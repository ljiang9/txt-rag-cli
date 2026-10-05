"""基于 TF-IDF 稀疏向量的余弦相似度检索。"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .index import TextIndex, Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class Retriever:
    def __init__(self, index: TextIndex):
        self.index = index

    def search(self, query: str, top_k: int = 3) -> list[ScoredChunk]:
        qv = self.index.query_vector(query)
        scored: list[ScoredChunk] = []
        for ch in self.index.chunks:
            cv = self.index.chunk_vector(ch)
            s = _cosine(qv, cv)
            if s > 0:
                scored.append(ScoredChunk(chunk=ch, score=s))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
