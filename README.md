# conversation-cluster-memory

A memory layer that keeps a long conversation's model context bounded without making earlier
facts unreachable. It sits beside whatever inference stack the application already uses: the
application keeps calling its own model, the library decides which text the model sees.

## Installation

```bash
uv sync
```

Torch is installed as a CPU-only wheel.

## Usage

```python
from conversation_cluster_memory import ConversationClusterMemory, ConversationKey

memory = ConversationClusterMemory(
    ConversationKey("conversation-42", user="user-7", tenant="tenant-1"),
    path="./memory",  # leave out to keep everything in memory
)
memory.add_message({"role": "user", "content": "Hello"})
messages = memory.window()  # the last `keep_recent_turns` turns, verbatim
```

- **Conversation key.** `ConversationKey(conversation, user=None, tenant=None)`, a plain string
  (a key with only a conversation), or any object with a `key_parts()` method that returns an
  ordered tuple of strings. Each conversation has its own SQLite rows and its own Chroma
  collection.
- **Turns.** A turn is one user message plus everything until the next user message.
  `keep_recent_turns` (default 10) counts turns. A turn is embedded once, when it leaves the
  recent window; turns whose embedding failed are backfilled on the next `add_message`.
- **Embedded text.** By default the user message, the assistant answer and, per tool result,
  only a header with the tool name and size. Pass `embedding_text=` to replace it. Tool results
  are always stored in full.
- **Storage.** With `path`, messages go to `archive.sqlite3` and vectors to `chroma/` in that
  directory. Without it, nothing touches disk.

Run the demo, which shows a built-in conversation and which turns have a vector:

```bash
uv run python -m conversation_cluster_memory
```

## Embedding model

The default is the local sentence-transformer
[`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
It was chosen because it is multilingual (50+ languages, so a conversation in any of them
clusters by topic), small and fast enough to run on CPU, and local: no message leaves the
machine and no API key is needed. It is downloaded on first use.

OpenAI embeddings work through the same LangChain `Embeddings` interface:

```python
from conversation_cluster_memory import openai_embeddings

memory = ConversationClusterMemory("conversation-42", embeddings=openai_embeddings())
```

`OPENAI_API_KEY` is read from the environment or from a `.env` file (copy `.env.example`). A
missing key raises `MissingOpenAIKeyError` only when an OpenAI component is used.

The store records the embedding model of every vector. Opening a conversation, or querying its
vectors, with another model raises `EmbeddingModelMismatchError`: two vector spaces are never
mixed.

With this model, sentences on the same topic often score only 0.5–0.7 cosine similarity, so
clustering thresholds must be tuned to it.
