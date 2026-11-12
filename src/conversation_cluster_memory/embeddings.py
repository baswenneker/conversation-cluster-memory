"""Embedding models and the text a turn is embedded as."""

import functools
import os
from collections.abc import Callable, Sequence

from dotenv import dotenv_values, find_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

type EmbeddingText = Callable[[Sequence[BaseMessage]], str]


class MissingOpenAIKeyError(RuntimeError):
    """An OpenAI component was requested but OPENAI_API_KEY is not set."""


@functools.cache
def default_embeddings() -> Embeddings:
    """The local multilingual sentence-transformer, loaded once per process (runs on CPU)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


def openai_embeddings(model: str = DEFAULT_OPENAI_EMBEDDING_MODEL) -> Embeddings:
    """OpenAI embeddings; OPENAI_API_KEY comes from the environment or a .env file."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model, api_key=openai_api_key())


def openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(find_dotenv(usecwd=True)).get(
        "OPENAI_API_KEY"
    )
    if not key:
        raise MissingOpenAIKeyError(
            "OPENAI_API_KEY is not set: put it in the environment or in a .env file "
            "(see .env.example)."
        )
    return key


def embedding_model_id(embeddings: Embeddings) -> str:
    """The id recorded with every vector, so vectors of two models are never mixed."""
    for attribute in ("model_name", "model", "model_id"):
        value = getattr(embeddings, attribute, None)
        if isinstance(value, str) and value:
            return value
    return f"{type(embeddings).__module__}.{type(embeddings).__qualname__}"


def default_embedding_text(messages: Sequence[BaseMessage]) -> str:
    """The user message, the assistant answers and, per tool result, only a header."""
    tool_names = {
        call["id"]: call["name"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.text}")
        elif isinstance(message, ToolMessage):
            name = message.name or tool_names.get(message.tool_call_id, "unknown tool")
            lines.append(f"Tool result {name}: {len(message.text)} characters")
        elif isinstance(message, AIMessage) and message.text:
            lines.append(f"Assistant: {message.text}")
    return "\n".join(lines)
