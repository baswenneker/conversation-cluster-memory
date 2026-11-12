import hashlib

import pytest
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic, offline embeddings that record every text they embed."""

    def __init__(self, model: str = "fake-model", size: int = 8, fail_times: int = 0) -> None:
        self.model = model
        self.size = size
        self.fail_times = fail_times
        self.texts: list[str] = []

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 + 0.01 for b in digest[: self.size]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("embedding service down")
        self.texts.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


class TopicEmbeddings(Embeddings):
    """Embeds a text as the vector of its topic: the part before ':' (e.g. "a: question 1")."""

    def __init__(self, topics: dict[str, list[float]]) -> None:
        self.topics = topics

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.topics[text.split(":")[0]]
