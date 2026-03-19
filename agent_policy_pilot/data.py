from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Passage:
    pid: str
    title: str
    text: str


@dataclass(frozen=True)
class HotpotExample:
    qid: str
    question: str
    answer: str
    context: list[tuple[str, list[str]]]


def load_hotpot_dev(path: str | Path) -> list[HotpotExample]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    examples: list[HotpotExample] = []
    for row in raw:
        context_items = [
            (title, sents)
            for title, sents in row.get("context", [])
            if isinstance(title, str) and isinstance(sents, list)
        ]
        examples.append(
            HotpotExample(
                qid=str(row.get("_id", "")),
                question=str(row.get("question", "")).strip(),
                answer=str(row.get("answer", "")).strip(),
                context=context_items,
            )
        )
    return examples


def sample_examples(examples: list[HotpotExample], sample_size: int, seed: int) -> list[HotpotExample]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(examples):
        raise ValueError(f"sample_size={sample_size} exceeds dataset size={len(examples)}")
    rng = random.Random(seed)
    idx = rng.sample(range(len(examples)), sample_size)
    return [examples[i] for i in idx]


def build_corpus_from_contexts(examples: Iterable[HotpotExample]) -> list[Passage]:
    passages: list[Passage] = []
    seen: set[tuple[str, str]] = set()
    for ex in examples:
        for title, sents in ex.context:
            text = " ".join(s.strip() for s in sents if s.strip()).strip()
            if not text:
                continue
            key = (title, text)
            if key in seen:
                continue
            seen.add(key)
            pid = f"{len(passages):08d}"
            passages.append(Passage(pid=pid, title=title, text=text))
    return passages


def load_passages_jsonl(path: str | Path) -> list[Passage]:
    path = Path(path)
    passages: list[Passage] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            pid = str(row.get("pid", "")).strip() or f"ext_{i:08d}"
            passages.append(Passage(pid=pid, title=title, text=text))
    if not passages:
        raise ValueError(f"No valid passages loaded from {path}")
    return passages
