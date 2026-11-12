import hashlib
import threading
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


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


class FakeSummaryModel(BaseChatModel):
    """An offline chat model that answers every summary request with a fixed structured output.

    It records every prompt. `fail_times` makes the first calls raise; `gate`, when set,
    blocks each call until the test sets it.
    """

    text: str = "summary"
    fail_times: int = 0
    gate: Any = None
    prompts: list[list[BaseMessage]] = []
    started: Any = None
    active: int = 0
    max_active: int = 0

    def model_post_init(self, context: Any) -> None:
        self.prompts = []
        self.started = threading.Event()

    @property
    def _llm_type(self) -> str:
        return "fake-summary"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeSummaryModel":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.prompts.append(messages)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.gate is not None:
                self.gate.wait(10)
            if self.fail_times:
                self.fail_times -= 1
                raise ConnectionError("chat model down")
        finally:
            self.active -= 1
        n = len(self.prompts)
        args = {
            "text": f"{self.text} {n}",
            "subject": f"subject {n}",
            "period": f"period {n}",
            "names": [f"name {n}"],
        }
        message = AIMessage(
            "",
            tool_calls=[{"name": "Summary", "args": args, "id": f"call-{n}"}],
            usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def summary_model() -> FakeSummaryModel:
    return FakeSummaryModel()
