"""
[INPUT] Fixed ICD disease names and normalized clinical text queries.
[OUTPUT] Deterministically ranked lexical candidates with decomposed scores.
[POS] CPU candidate-recall baseline and shared first stage for neural/graph rerankers.
[UPDATE] Keep retrieval configuration, score semantics and experiment docs synchronized.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from mednorm.text import normalize_label, normalize_text


def dice_similarity(left: str, right: str) -> float:
    """Character multiset Dice similarity used as an interpretable reranking feature."""
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    intersection = sum((left_counts & right_counts).values())
    return 2.0 * intersection / (len(left) + len(right))


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    name: str
    score: float
    tfidf_score: float
    dice_score: float


class CharacterTfidfRetriever:
    """Sparse character TF-IDF retrieval followed by lightweight Dice reranking."""

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (1, 3),
        dice_weight: float = 0.2,
        rerank_pool_size: int = 300,
    ) -> None:
        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError("dice_weight must be between zero and one")
        if rerank_pool_size < 1:
            raise ValueError("rerank_pool_size must be positive")
        self.ngram_range = ngram_range
        self.dice_weight = dice_weight
        self.rerank_pool_size = rerank_pool_size
        self._vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            lowercase=False,
            dtype=np.float32,
            norm="l2",
        )
        self._names: tuple[str, ...] = ()
        self._matrix: csr_matrix | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def fit(self, names: Sequence[str]) -> CharacterTfidfRetriever:
        normalized_names = tuple(dict.fromkeys(normalize_label(name) for name in names))
        if not normalized_names or any(not name for name in normalized_names):
            raise ValueError("at least one non-empty candidate name is required")
        self._names = normalized_names
        self._matrix = self._vectorizer.fit_transform(normalized_names).tocsr()
        return self

    def search_one(self, query: str, *, k: int = 200) -> tuple[RankedCandidate, ...]:
        if self._matrix is None:
            raise RuntimeError("retriever must be fit before search")
        if k < 1:
            raise ValueError("k must be positive")

        normalized_query = normalize_text(query)
        query_vector = self._vectorizer.transform([normalized_query])
        tfidf_scores = (query_vector @ self._matrix.T).toarray().ravel()
        result_count = min(k, len(self._names))
        pool_count = min(max(result_count, self.rerank_pool_size), len(self._names))

        if pool_count == len(self._names):
            pool_indices = np.arange(len(self._names))
        else:
            split_at = len(tfidf_scores) - pool_count
            pool_indices = np.argpartition(tfidf_scores, split_at)[split_at:]

        candidates: list[tuple[int, float, float]] = []
        for raw_index in pool_indices:
            index = int(raw_index)
            tfidf_score = float(tfidf_scores[index])
            dice_score = dice_similarity(normalized_query, self._names[index])
            score = (1.0 - self.dice_weight) * tfidf_score + self.dice_weight * dice_score
            candidates.append((index, score, dice_score))

        candidates.sort(key=lambda item: (-item[1], item[0]))
        return tuple(
            RankedCandidate(
                name=self._names[index],
                score=score,
                tfidf_score=float(tfidf_scores[index]),
                dice_score=dice_score,
            )
            for index, score, dice_score in candidates[:result_count]
        )

    def search(
        self, queries: Sequence[str], *, k: int = 200
    ) -> tuple[tuple[RankedCandidate, ...], ...]:
        return tuple(self.search_one(query, k=k) for query in queries)

