"""Vectors of one conversation (turns, cluster centroids), in Chroma collections of its own."""

import chromadb
import numpy as np
from chromadb.api import ClientAPI


class EmbeddingModelMismatchError(ValueError):
    """The archive was built with another embedding model than the one in use."""


class VectorStore:
    """One conversation's Chroma collection. Every vector records the model that made it."""

    def __init__(self, client: ClientAPI, name: str, model_id: str) -> None:
        self.model_id = model_id
        self._collection = client.get_or_create_collection(
            name,
            metadata={"embedding_model": model_id},
            configuration={"hnsw": {"space": "cosine"}},
        )
        self.check(self._collection.metadata.get("embedding_model"))

    def check(self, model_id: object) -> None:
        if model_id != self.model_id:
            raise EmbeddingModelMismatchError(
                f"this conversation's vectors were made with {model_id!r}, not "
                f"{self.model_id!r}; re-index the archive before switching models"
            )

    def turns(self) -> set[int]:
        """The ids (turns, or clusters for a centroid store) that have a vector."""
        return {int(i) for i in self._collection.get(include=[])["ids"]}

    def get(self, ids: list[int] | None = None) -> dict[int, list[float]]:
        """The vectors with these ids (all when None)."""
        if ids == []:
            return {}
        result = self._collection.get(
            ids=None if ids is None else [str(i) for i in ids], include=["embeddings"]
        )
        embeddings = result["embeddings"]
        assert embeddings is not None
        return {
            int(i): np.asarray(v, dtype=float).tolist()
            for i, v in zip(result["ids"], embeddings, strict=True)
        }

    def add(self, vectors: dict[int, list[float]]) -> None:
        self._collection.upsert(
            ids=[str(i) for i in vectors],
            embeddings=list(vectors.values()),
            metadatas=[{"id": i, "embedding_model": self.model_id} for i in vectors],
        )

    def query(self, vector: list[float], k: int, model_id: str) -> list[tuple[int, float]]:
        """The `k` nearest turns as (turn, cosine similarity), most similar first."""
        self.check(model_id)
        result = self._collection.query(query_embeddings=[vector], n_results=k)
        ids, distances = result["ids"][0], (result["distances"] or [[]])[0]
        return [(int(i), 1 - d) for i, d in zip(ids, distances, strict=True)]


def chroma_client(path: str | None) -> ClientAPI:
    settings = chromadb.config.Settings(anonymized_telemetry=False)
    if path is None:
        return chromadb.EphemeralClient(settings=settings)
    return chromadb.PersistentClient(path=path, settings=settings)
