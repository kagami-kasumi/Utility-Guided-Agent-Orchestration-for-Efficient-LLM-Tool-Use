from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .data import HotpotExample
from .llm import OpenAIChatModel, parse_json_maybe
from .search import BM25Searcher, RetrievalHit


def _format_hits(hits: list[RetrievalHit], max_chars: int = 1800) -> str:
    if not hits:
        return "(no passages)"
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        text = hit.text.replace("\n", " ").strip()
        parts.append(f"[{i}] {hit.title} | score={hit.score:.3f}\n{text}")
    text = "\n\n".join(parts)
    return text[:max_chars]


def _dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    seen: set[str] = set()
    unique: list[RetrievalHit] = []
    for hit in hits:
        if hit.pid in seen:
            continue
        seen.add(hit.pid)
        unique.append(hit)
    return unique


def _to_float_01(value: Any, default: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, x))


@dataclass
class PolicyResult:
    prediction: str
    token_total: int
    token_prompt: int
    token_completion: int
    tool_calls: int
    trajectory: list[dict[str, Any]]


class BasePolicy:
    name = "base"

    def __init__(
        self,
        max_tool_calls: int,
        topk: int,
        token_budget: int | None = None,
        max_completion_tokens: int = 64,
        max_decision_tokens: int = 96,
    ):
        self.max_tool_calls = max_tool_calls
        self.topk = topk
        self.token_budget = token_budget
        self.max_completion_tokens = max_completion_tokens
        self.max_decision_tokens = max_decision_tokens

    def run(self, ex: HotpotExample, model: OpenAIChatModel, searcher: BM25Searcher) -> PolicyResult:
        raise NotImplementedError

    def _over_budget(self, consumed_tokens: int) -> bool:
        return self.token_budget is not None and consumed_tokens >= self.token_budget

    def _chat(
        self,
        model: OpenAIChatModel,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ):
        return model.chat(messages, response_format=response_format, max_tokens=max_tokens)


class DirectPolicy(BasePolicy):
    name = "direct"

    def run(self, ex: HotpotExample, model: OpenAIChatModel, searcher: BM25Searcher) -> PolicyResult:
        del searcher
        traj: list[dict[str, Any]] = []
        messages = [
            {"role": "system", "content": "Answer the question with a short factual span."},
            {"role": "user", "content": ex.question},
        ]
        resp = self._chat(model, messages, max_tokens=self.max_completion_tokens)
        traj.append(
            {
                "type": "model",
                "step": "direct_answer",
                "question": ex.question,
                "response": resp.text,
                "usage": resp.usage.__dict__,
            }
        )
        return PolicyResult(
            prediction=resp.text,
            token_total=resp.usage.total_tokens,
            token_prompt=resp.usage.prompt_tokens,
            token_completion=resp.usage.completion_tokens,
            tool_calls=0,
            trajectory=traj,
        )


class WorkflowSearchOncePolicy(BasePolicy):
    name = "workflow_search_once"

    def run(self, ex: HotpotExample, model: OpenAIChatModel, searcher: BM25Searcher) -> PolicyResult:
        traj: list[dict[str, Any]] = []
        token_prompt = 0
        token_completion = 0

        hits = searcher.search(ex.question, self.topk)
        traj.append(
            {
                "type": "tool",
                "step": "search_once",
                "query": ex.question,
                "topk": self.topk,
                "hits": [hit.__dict__ for hit in hits],
            }
        )
        evidence = _format_hits(hits)
        messages = [
            {"role": "system", "content": "Use the retrieved passages to answer the question with a short factual span."},
            {"role": "user", "content": f"Question:\n{ex.question}\n\nPassages:\n{evidence}\n\nAnswer:"},
        ]
        resp = self._chat(model, messages, max_tokens=self.max_completion_tokens)
        token_prompt += resp.usage.prompt_tokens
        token_completion += resp.usage.completion_tokens
        traj.append(
            {
                "type": "model",
                "step": "final_answer",
                "response": resp.text,
                "usage": resp.usage.__dict__,
            }
        )
        return PolicyResult(
            prediction=resp.text,
            token_total=token_prompt + token_completion,
            token_prompt=token_prompt,
            token_completion=token_completion,
            tool_calls=1,
            trajectory=traj,
        )


class WorkflowSearchTwicePolicy(BasePolicy):
    name = "workflow_search_twice"

    def run(self, ex: HotpotExample, model: OpenAIChatModel, searcher: BM25Searcher) -> PolicyResult:
        traj: list[dict[str, Any]] = []
        token_prompt = 0
        token_completion = 0

        hits1 = searcher.search(ex.question, self.topk)
        traj.append(
            {
                "type": "tool",
                "step": "search_once",
                "query": ex.question,
                "topk": self.topk,
                "hits": [hit.__dict__ for hit in hits1],
            }
        )
        title_terms = " ".join(hit.title for hit in hits1[:2] if hit.title.strip())
        query2 = f"{ex.question} {title_terms}".strip() if title_terms else ex.question
        hits2 = searcher.search(query2, self.topk)
        traj.append(
            {
                "type": "tool",
                "step": "search_twice",
                "query": query2,
                "topk": self.topk,
                "hits": [hit.__dict__ for hit in hits2],
            }
        )
        evidence = _format_hits(_dedupe_hits(hits1 + hits2), max_chars=2200)
        messages = [
            {"role": "system", "content": "Use the retrieved passages to answer the question with a short factual span."},
            {"role": "user", "content": f"Question:\n{ex.question}\n\nPassages:\n{evidence}\n\nAnswer:"},
        ]
        resp = self._chat(model, messages, max_tokens=self.max_completion_tokens)
        token_prompt += resp.usage.prompt_tokens
        token_completion += resp.usage.completion_tokens
        traj.append(
            {
                "type": "model",
                "step": "final_answer",
                "response": resp.text,
                "usage": resp.usage.__dict__,
            }
        )
        return PolicyResult(
            prediction=resp.text,
            token_total=token_prompt + token_completion,
            token_prompt=token_prompt,
            token_completion=token_completion,
            tool_calls=2,
            trajectory=traj,
        )


class WorkflowSearchVerifyPolicy(BasePolicy):
    name = "workflow_search_verify"

    def run(self, ex: HotpotExample, model: OpenAIChatModel, searcher: BM25Searcher) -> PolicyResult:
        traj: list[dict[str, Any]] = []
        token_prompt = 0
        token_completion = 0

        hits1 = searcher.search(ex.question, self.topk)
        traj.append(
            {
                "type": "tool",
                "step": "search_once",
                "query": ex.question,
                "topk": self.topk,
                "hits": [hit.__dict__ for hit in hits1],
            }
        )
        evidence1 = _format_hits(hits1, max_chars=1600)
        draft_messages = [
            {"role": "system", "content": "Draft a short factual answer from the retrieved passages."},
            {"role": "user", "content": f"Question:\n{ex.question}\n\nPassages:\n{evidence1}\n\nDraft answer:"},
        ]
        draft_resp = self._chat(model, draft_messages, max_tokens=self.max_completion_tokens)
        token_prompt += draft_resp.usage.prompt_tokens
        token_completion += draft_resp.usage.completion_tokens
        draft_answer = draft_resp.text.strip() or "unknown"
        traj.append(
            {
                "type": "model",
                "step": "draft_answer",
                "response": draft_resp.text,
                "usage": draft_resp.usage.__dict__,
            }
        )

        verify_query = f"{ex.question} {draft_answer}".strip()
        hits2 = searcher.search(verify_query, self.topk)
        traj.append(
            {
                "type": "tool",
                "step": "verify_search",
                "query": verify_query,
                "topk": self.topk,
                "hits": [hit.__dict__ for hit in hits2],
            }
        )
        evidence = _format_hits(_dedupe_hits(hits1 + hits2), max_chars=2200)
        final_messages = [
            {
                "role": "system",
                "content": "Verify the draft answer against the retrieved passages and return the best short factual span.",
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{ex.question}\n\n"
                    f"Draft answer:\n{draft_answer}\n\n"
                    f"Passages:\n{evidence}\n\n"
                    "Final answer:"
                ),
            },
        ]
        final_resp = self._chat(model, final_messages, max_tokens=self.max_completion_tokens)
        token_prompt += final_resp.usage.prompt_tokens
        token_completion += final_resp.usage.completion_tokens
        traj.append(
            {
                "type": "model",
                "step": "final_answer",
                "response": final_resp.text,
                "usage": final_resp.usage.__dict__,
            }
        )
        return PolicyResult(
            prediction=final_resp.text,
            token_total=token_prompt + token_completion,
            token_prompt=token_prompt,
            token_completion=token_completion,
            tool_calls=2,
            trajectory=traj,
        )


class ReActPolicy(BasePolicy):
    name = "react"

    def run(self, ex: HotpotExample, model: OpenAIChatModel, searcher: BM25Searcher) -> PolicyResult:
        traj: list[dict[str, Any]] = []
        evidence_blocks: list[str] = []
        token_prompt = 0
        token_completion = 0
        tool_calls = 0
        answer = ""

        for t in range(self.max_tool_calls + 1):
            if self._over_budget(token_prompt + token_completion):
                traj.append({"type": "control", "step": f"budget_stop_{t}"})
                break

            obs = "\n\n".join(evidence_blocks) if evidence_blocks else "(none)"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a ReAct agent. Return only JSON with keys: "
                        "action (search|answer), query, answer, confidence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {ex.question}\n\n"
                        f"Current observations:\n{obs}\n\n"
                        "If information is insufficient, action=search and provide a query. "
                        "If sufficient, action=answer and provide answer."
                    ),
                },
            ]
            resp = self._chat(
                model,
                messages,
                response_format={"type": "json_object"},
                max_tokens=self.max_decision_tokens,
            )
            token_prompt += resp.usage.prompt_tokens
            token_completion += resp.usage.completion_tokens
            action_obj = parse_json_maybe(resp.text)
            action = str(action_obj.get("action", "answer")).strip().lower()
            traj.append(
                {
                    "type": "model",
                    "step": f"react_decision_{t}",
                    "response": resp.text,
                    "parsed": action_obj,
                    "usage": resp.usage.__dict__,
                }
            )

            if action == "search" and tool_calls < self.max_tool_calls:
                query = str(action_obj.get("query", "")).strip() or ex.question
                hits = searcher.search(query, self.topk)
                tool_calls += 1
                evidence = _format_hits(hits, max_chars=1400)
                evidence_blocks.append(f"Search query: {query}\n{evidence}")
                traj.append(
                    {
                        "type": "tool",
                        "step": f"search_{tool_calls}",
                        "query": query,
                        "hits": [hit.__dict__ for hit in hits],
                    }
                )
                continue

            answer = str(action_obj.get("answer", "")).strip()
            if answer:
                break

        if not answer:
            if self._over_budget(token_prompt + token_completion):
                answer = "unknown"
            else:
                obs = "\n\n".join(evidence_blocks) if evidence_blocks else "(none)"
                final_messages = [
                    {"role": "system", "content": "Answer with a short factual span."},
                    {"role": "user", "content": f"Question: {ex.question}\n\nObservations:\n{obs}\n\nAnswer:"},
                ]
                final_resp = self._chat(model, final_messages, max_tokens=self.max_completion_tokens)
                token_prompt += final_resp.usage.prompt_tokens
                token_completion += final_resp.usage.completion_tokens
                answer = final_resp.text
                traj.append(
                    {
                        "type": "model",
                        "step": "react_fallback_answer",
                        "response": final_resp.text,
                        "usage": final_resp.usage.__dict__,
                    }
                )

        return PolicyResult(
            prediction=answer,
            token_total=token_prompt + token_completion,
            token_prompt=token_prompt,
            token_completion=token_completion,
            tool_calls=tool_calls,
            trajectory=traj,
        )


class ThresholdPolicy(BasePolicy):
    name = "threshold"

    def __init__(
        self,
        max_tool_calls: int,
        topk: int,
        threshold: float = 0.15,
        cost_weight: float = 0.35,
        **kwargs: Any,
    ):
        super().__init__(max_tool_calls=max_tool_calls, topk=topk, **kwargs)
        self.threshold = threshold
        self.cost_weight = cost_weight

    def _step_cost(self, current_tool_calls: int) -> float:
        if self.max_tool_calls <= 0:
            return 1.0
        return (current_tool_calls + 1) / self.max_tool_calls

    def run(self, ex: HotpotExample, model: OpenAIChatModel, searcher: BM25Searcher) -> PolicyResult:
        traj: list[dict[str, Any]] = []
        evidence_blocks: list[str] = []
        token_prompt = 0
        token_completion = 0
        tool_calls = 0
        draft_answer = ""

        for t in range(self.max_tool_calls + 1):
            if self._over_budget(token_prompt + token_completion):
                traj.append({"type": "control", "step": f"budget_stop_{t}"})
                break

            obs = "\n\n".join(evidence_blocks) if evidence_blocks else "(none)"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Return only JSON with keys: uncertainty, expected_gain, query, draft_answer. "
                        "uncertainty and expected_gain are in [0,1]."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {ex.question}\n\n"
                        f"Current observations:\n{obs}\n\n"
                        "Estimate uncertainty of your current draft answer and expected gain from one more search."
                    ),
                },
            ]
            resp = self._chat(
                model,
                messages,
                response_format={"type": "json_object"},
                max_tokens=self.max_decision_tokens,
            )
            token_prompt += resp.usage.prompt_tokens
            token_completion += resp.usage.completion_tokens

            payload = parse_json_maybe(resp.text)
            uncertainty = _to_float_01(payload.get("uncertainty"), default=0.5)
            expected_gain = _to_float_01(payload.get("expected_gain"), default=0.3)
            draft_answer = str(payload.get("draft_answer", "")).strip() or draft_answer
            query = str(payload.get("query", "")).strip() or ex.question
            step_cost = self._step_cost(tool_calls)
            score = uncertainty * expected_gain - self.cost_weight * step_cost
            will_search = score >= self.threshold and tool_calls < self.max_tool_calls

            traj.append(
                {
                    "type": "model",
                    "step": f"threshold_eval_{t}",
                    "response": resp.text,
                    "parsed": payload,
                    "decision": {
                        "uncertainty": uncertainty,
                        "expected_gain": expected_gain,
                        "step_cost": step_cost,
                        "score": score,
                        "threshold": self.threshold,
                        "will_search": will_search,
                    },
                    "usage": resp.usage.__dict__,
                }
            )

            if will_search:
                hits = searcher.search(query, self.topk)
                tool_calls += 1
                evidence = _format_hits(hits, max_chars=1400)
                evidence_blocks.append(f"Search query: {query}\n{evidence}")
                traj.append(
                    {
                        "type": "tool",
                        "step": f"search_{tool_calls}",
                        "query": query,
                        "hits": [hit.__dict__ for hit in hits],
                    }
                )
                continue
            break

        answer = draft_answer
        if not answer or answer.lower() in {"unknown", "not sure", "i don't know"}:
            if self._over_budget(token_prompt + token_completion):
                answer = "unknown"
            else:
                obs = "\n\n".join(evidence_blocks) if evidence_blocks else "(none)"
                final_messages = [
                    {"role": "system", "content": "Answer with a short factual span. If unknown, output best guess."},
                    {"role": "user", "content": f"Question: {ex.question}\n\nObservations:\n{obs}\n\nAnswer:"},
                ]
                final_resp = self._chat(model, final_messages, max_tokens=self.max_completion_tokens)
                token_prompt += final_resp.usage.prompt_tokens
                token_completion += final_resp.usage.completion_tokens
                answer = final_resp.text
                traj.append(
                    {
                        "type": "model",
                        "step": "threshold_finalize",
                        "response": final_resp.text,
                        "usage": final_resp.usage.__dict__,
                    }
                )

        answer = answer.strip() or "unknown"
        return PolicyResult(
            prediction=answer,
            token_total=token_prompt + token_completion,
            token_prompt=token_prompt,
            token_completion=token_completion,
            tool_calls=tool_calls,
            trajectory=traj,
        )


class UtilityPolicy(BasePolicy):
    name = "policy"

    def __init__(
        self,
        max_tool_calls: int,
        topk: int,
        lambda_cost: float = 0.3,
        lambda_uncertainty: float = 0.2,
        lambda_redundancy: float = 0.2,
        cost_mode: str = "step",
        redundancy_mode: str = "exact",
        token_cost_reference: float = 1200.0,
        latency_cost_reference: float = 1.0,
        semantic_redundancy_threshold: float = 0.5,
        use_expected_gain: bool = True,
        use_uncertainty: bool = True,
        use_redundancy: bool = True,
        use_stop_policy: bool = True,
        **kwargs: Any,
    ):
        super().__init__(max_tool_calls=max_tool_calls, topk=topk, **kwargs)
        self.lambda_cost = lambda_cost
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_redundancy = lambda_redundancy
        self.cost_mode = cost_mode
        self.redundancy_mode = redundancy_mode
        self.token_cost_reference = max(1.0, token_cost_reference)
        self.latency_cost_reference = max(1e-6, latency_cost_reference)
        self.semantic_redundancy_threshold = semantic_redundancy_threshold
        self.use_expected_gain = use_expected_gain
        self.use_uncertainty = use_uncertainty
        self.use_redundancy = use_redundancy
        self.use_stop_policy = use_stop_policy

    def _step_cost(self, current_tool_calls: int) -> float:
        if self.max_tool_calls <= 0:
            return 1.0
        return (current_tool_calls + 1) / self.max_tool_calls

    @staticmethod
    def _exact_redundancy(query: str, prev_queries: list[str]) -> float:
        q = query.strip().lower()
        if not q or not prev_queries:
            return 0.0
        return 1.0 if q in prev_queries else 0.0

    def _cost_value(self, current_tool_calls: int, consumed_tokens: int, elapsed_time: float) -> float:
        if self.cost_mode == "token":
            return min(1.0, consumed_tokens / self.token_cost_reference)
        if self.cost_mode == "latency":
            return min(1.0, elapsed_time / self.latency_cost_reference)
        return self._step_cost(current_tool_calls)

    @staticmethod
    def _semantic_redundancy_from_hits(
        hits: list[RetrievalHit],
        previous_hit_sets: list[set[str]],
    ) -> float:
        if not hits or not previous_hit_sets:
            return 0.0
        hit_ids = {hit.pid for hit in hits if hit.pid}
        if not hit_ids:
            return 0.0
        return max(len(hit_ids & prev_ids) / len(hit_ids) for prev_ids in previous_hit_sets)

    def run(self, ex: HotpotExample, model: OpenAIChatModel, searcher: BM25Searcher) -> PolicyResult:
        traj: list[dict[str, Any]] = []
        evidence_blocks: list[str] = []
        previous_queries: list[str] = []
        previous_hit_sets: list[set[str]] = []
        token_prompt = 0
        token_completion = 0
        tool_calls = 0
        draft_answer = ""
        start_time = time.perf_counter()

        for t in range(self.max_tool_calls + 1):
            if self._over_budget(token_prompt + token_completion):
                traj.append({"type": "control", "step": f"budget_stop_{t}"})
                break

            obs = "\n\n".join(evidence_blocks) if evidence_blocks else "(none)"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Return only JSON with keys: uncertainty, expected_gain, query, draft_answer. "
                        "uncertainty and expected_gain are in [0,1]."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {ex.question}\n\n"
                        f"Current observations:\n{obs}\n\n"
                        "Estimate uncertainty, expected gain from one additional search, and a draft answer."
                    ),
                },
            ]
            resp = self._chat(
                model,
                messages,
                response_format={"type": "json_object"},
                max_tokens=self.max_decision_tokens,
            )
            token_prompt += resp.usage.prompt_tokens
            token_completion += resp.usage.completion_tokens
            payload = parse_json_maybe(resp.text)

            uncertainty = _to_float_01(payload.get("uncertainty"), default=0.5)
            expected_gain = _to_float_01(payload.get("expected_gain"), default=0.4)
            query = str(payload.get("query", "")).strip() or ex.question
            draft_answer = str(payload.get("draft_answer", "")).strip() or draft_answer

            preview_hits: list[RetrievalHit] | None = None
            if self.use_redundancy and self.redundancy_mode == "semantic":
                preview_hits = searcher.search(query, self.topk)
                redundancy = self._semantic_redundancy_from_hits(preview_hits, previous_hit_sets)
            else:
                redundancy = self._exact_redundancy(query, previous_queries) if self.use_redundancy else 0.0

            step_cost = self._cost_value(
                current_tool_calls=tool_calls,
                consumed_tokens=token_prompt + token_completion,
                elapsed_time=time.perf_counter() - start_time,
            )
            gain_term = expected_gain if self.use_expected_gain else 1.0
            uncertainty_term = uncertainty if self.use_uncertainty else 0.0
            utility = (
                gain_term
                - self.lambda_cost * step_cost
                - self.lambda_uncertainty * uncertainty_term
                - self.lambda_redundancy * redundancy
            )
            should_search = tool_calls < self.max_tool_calls and (utility > 0.0 or not self.use_stop_policy)

            traj.append(
                {
                    "type": "model",
                    "step": f"policy_eval_{t}",
                    "response": resp.text,
                    "parsed": payload,
                    "decision": {
                        "utility": utility,
                        "expected_gain": expected_gain,
                        "uncertainty": uncertainty,
                        "redundancy": redundancy,
                        "step_cost": step_cost,
                        "cost_mode": self.cost_mode,
                        "redundancy_mode": self.redundancy_mode,
                        "should_search": should_search,
                    },
                    "usage": resp.usage.__dict__,
                }
            )

            if should_search:
                hits = preview_hits if preview_hits is not None else searcher.search(query, self.topk)
                tool_calls += 1
                previous_queries.append(query.strip().lower())
                previous_hit_sets.append({hit.pid for hit in hits if hit.pid})
                evidence = _format_hits(hits, max_chars=1400)
                evidence_blocks.append(f"Search query: {query}\n{evidence}")
                traj.append(
                    {
                        "type": "tool",
                        "step": f"search_{tool_calls}",
                        "query": query,
                        "redundancy_score": redundancy,
                        "redundancy_mode": self.redundancy_mode,
                        "is_redundant": redundancy >= self.semantic_redundancy_threshold,
                        "hits": [hit.__dict__ for hit in hits],
                    }
                )
                continue
            break

        answer = draft_answer
        if not answer or answer.lower() in {"unknown", "not sure", "i don't know"}:
            if self._over_budget(token_prompt + token_completion):
                answer = "unknown"
            else:
                obs = "\n\n".join(evidence_blocks) if evidence_blocks else "(none)"
                final_messages = [
                    {"role": "system", "content": "Answer with a short factual span. If unknown, output best guess."},
                    {"role": "user", "content": f"Question: {ex.question}\n\nObservations:\n{obs}\n\nAnswer:"},
                ]
                final_resp = self._chat(model, final_messages, max_tokens=self.max_completion_tokens)
                token_prompt += final_resp.usage.prompt_tokens
                token_completion += final_resp.usage.completion_tokens
                answer = final_resp.text
                traj.append(
                    {
                        "type": "model",
                        "step": "policy_finalize",
                        "response": final_resp.text,
                        "usage": final_resp.usage.__dict__,
                    }
                )

        answer = answer.strip() or "unknown"
        return PolicyResult(
            prediction=answer,
            token_total=token_prompt + token_completion,
            token_prompt=token_prompt,
            token_completion=token_completion,
            tool_calls=tool_calls,
            trajectory=traj,
        )
