from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: TokenUsage
    raw: Any


def _estimate_tokens(text: str, model: str) -> int:
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
    return max(1, len(text.split()))


def _estimate_usage(messages: list[dict[str, str]], completion: str, model: str) -> TokenUsage:
    prompt_text = "\n".join(msg.get("content", "") for msg in messages)
    prompt_tokens = _estimate_tokens(prompt_text, model)
    completion_tokens = _estimate_tokens(completion, model)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


class OpenAIChatModel:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        seed: int | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = self.client.chat.completions.create(**payload)
        text = (resp.choices[0].message.content or "").strip()
        if resp.usage is not None:
            usage = TokenUsage(
                prompt_tokens=int(resp.usage.prompt_tokens or 0),
                completion_tokens=int(resp.usage.completion_tokens or 0),
                total_tokens=int(resp.usage.total_tokens or 0),
            )
        else:
            usage = _estimate_usage(messages, text, self.model)
        return LLMResponse(text=text, usage=usage, raw=resp)


def parse_json_maybe(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}
