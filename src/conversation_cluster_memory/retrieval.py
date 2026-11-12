"""Retrieval hits and how vector and keyword results are merged into one ranking."""

from collections.abc import Sequence
from dataclasses import dataclass

# The usual constant of reciprocal rank fusion: damps the weight of the very first ranks.
RRF_K = 60

REWRITE_PROMPT = """\
Rewrite the user's message as a short search query for earlier topics of this conversation. \
Keep names, identifiers, numbers and dates literally. Return only the query."""


@dataclass(frozen=True)
class RetrievalHit:
    """A cluster found for a query: its summary or, if not yet summarized, its original turns."""

    cluster: int
    similarity: float  # cosine similarity of the query to the cluster's vector
    start: str  # ISO time of the cluster's first message
    end: str  # ISO time of its last message
    is_summary: bool
    content: str


def hit_text(hit: RetrievalHit) -> str:
    """The hit as the model reads it, in the window or as a tool result."""
    kind = "summary" if hit.is_summary else "original turns"
    return f"Retrieved from memory ({kind}, {hit.start} to {hit.end}):\n{hit.content}"


def fuse(*rankings: Sequence[int]) -> list[int]:
    """Reciprocal rank fusion: one ranking from several, so a cluster high in either counts.

    Scores of the vector and the keyword search are on different scales; ranks are not.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cluster in enumerate(ranking, start=1):
            scores[cluster] = scores.get(cluster, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda cluster: (-scores[cluster], cluster))
