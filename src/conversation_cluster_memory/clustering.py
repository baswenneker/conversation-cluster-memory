"""Batch clustering of turns by topic."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Float slack so a similarity of exactly the threshold still counts as reaching it.
_EPSILON = 1e-9


@dataclass(frozen=True)
class Cluster:
    """A set of turns about one topic."""

    id: int
    turns: tuple[int, ...]
    oldest_at: str  # ISO time of the oldest member turn's first message
    centroid: list[float]


class ClusteringStrategy(Protocol):
    """Splits a batch of turns into groups; the same input must always give the same groups."""

    def cluster(
        self, turns: Sequence[int], vectors: Sequence[Sequence[float]], threshold: float
    ) -> list[list[int]]:
        """Groups of turn numbers, each sorted, ordered by their first turn."""
        ...


class AgglomerativeClustering:
    """Average-linkage clustering on cosine similarity, cut at `1 - threshold`.

    Turns close in time get `proximity_bonus / distance in turns` added to their similarity,
    so neighbours on one topic stay together.
    """

    def __init__(self, proximity_bonus: float = 0.05) -> None:
        self.proximity_bonus = proximity_bonus

    def cluster(
        self, turns: Sequence[int], vectors: Sequence[Sequence[float]], threshold: float
    ) -> list[list[int]]:
        if len(turns) == 1:
            return [list(turns)]
        unit = _normalize(np.asarray(vectors, dtype=float))
        similarity = unit @ unit.T
        numbers = np.asarray(turns, dtype=float)
        gap = np.abs(numbers[:, None] - numbers[None, :])
        np.fill_diagonal(gap, 1)  # the diagonal is dropped below; avoids dividing by zero
        distance = np.clip(1 - (similarity + self.proximity_bonus / gap), 0, None)
        np.fill_diagonal(distance, 0)
        tree = linkage(squareform(distance, checks=False), method="average")
        labels = fcluster(tree, t=1 - threshold + _EPSILON, criterion="distance")
        groups: dict[int, list[int]] = {}
        for turn, label in sorted(zip(turns, labels, strict=True)):
            groups.setdefault(int(label), []).append(turn)
        return sorted(groups.values())


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    """The normalized mean of the normalized vectors."""
    return _normalize(_normalize(np.asarray(vectors, dtype=float)).mean(axis=0)).tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(_normalize(np.asarray(a, dtype=float)) @ _normalize(np.asarray(b, dtype=float)))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
