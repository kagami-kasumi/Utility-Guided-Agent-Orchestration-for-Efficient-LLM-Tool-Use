from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .data import Passage


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class RetrievalHit:
    pid: str
    title: str
    text: str
    score: float


class BM25Searcher:
    def __init__(self, passages: list[Passage]):
        if not passages:
            raise ValueError("passages is empty; cannot build BM25 index")
        self._passages = passages
        self._corpus_tokens = [_tokenize(f"{p.title} {p.text}") for p in passages]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, topk: int) -> list[RetrievalHit]:
        topk = max(1, topk)
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [
            RetrievalHit(
                pid=self._passages[i].pid,
                title=self._passages[i].title,
                text=self._passages[i].text,
                score=float(scores[i]),
            )
            for i in top_idx
        ]

