from pathlib import Path

import pytest

from conversation_cluster_memory import (
    ConversationClusterMemory,
    MissingOpenAIKeyError,
    openai_embeddings,
)


@pytest.fixture
def no_key(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_missing_key_raises_a_clear_error_when_openai_is_used(no_key: Path) -> None:
    with pytest.raises(MissingOpenAIKeyError, match="OPENAI_API_KEY"):
        openai_embeddings()


def test_a_missing_key_does_not_matter_when_openai_is_not_used(no_key: Path, embeddings) -> None:
    ConversationClusterMemory("c1", embeddings=embeddings).add_message("hi")


def test_the_key_is_read_from_dotenv(no_key: Path) -> None:
    (no_key / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n")

    model = openai_embeddings()

    assert model.openai_api_key.get_secret_value() == "sk-from-dotenv"  # ty: ignore[missing-attribute]


def test_openai_embeddings_are_selectable(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    memory = ConversationClusterMemory("c1", embeddings=openai_embeddings())

    assert memory.vectors.model_id == "text-embedding-3-small"
