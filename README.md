# conversation-cluster-memory

A memory layer that keeps a chat model's context bounded in conversations that last months, without making earlier facts unreachable.

`ConversationClusterMemory` sits beside whatever inference stack your application already uses. Your application keeps calling its own model; the library decides which text the model sees:

- every message is stored per conversation in a persistent archive, and compression never deletes from it;
- the most recent turns stay verbatim in the window;
- older turns are embedded, clustered by topic in batches, and each cluster is replaced in the window by a summary;
- on a new question, hybrid retrieval (vector plus keyword search) brings relevant old clusters back into the window within a token budget;
- an automatic evaluation measures how many tokens compaction saves and whether answers stay correct.

The full design, every decision behind it, and a framework-agnostic specification you can hand to another coding agent are in [`docs/plan.md`](docs/plan.md). Terms such as *turn*, *window* and *cluster* are defined in the [glossary](GLOSSARY.md).

## Explainer video

> **Placeholder.** A short explainer video will appear here. It will be recorded over the [motion graph](docs/motion-graph/index.html) (`docs/motion-graph/index.html`), a click-through page that shows embedding, clustering, summarization and retrieval step by step. The page arrives with its own feature; once it is there, open it locally in a browser to walk through the concept yourself.

## Installation

[uv](https://docs.astral.sh/uv/) is the only supported install path; there is no pip requirements file. The project needs Python 3.13, which uv installs for you.

```bash
uv sync                                       # create .venv and install all dependencies
uv run pytest                                 # run the test suite
uv run python -m conversation_cluster_memory  # run the demo on a built-in conversation, in memory
```

Tests need no API key and no network. The default embedding model is downloaded once, the first time it is used. OpenAI components (OpenAI embeddings, the summary model, the evaluation judge) read `OPENAI_API_KEY` from a `.env` file; copy `.env.example` to `.env` and fill it in. Without a key, using such a component raises a clear error.

## Default embedding model

The default embedding model is [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), a local sentence-transformer. It was chosen because:

- it runs locally, so conversation text never leaves your machine to be embedded;
- it is multilingual, so conversations in languages other than English work without changing the model;
- it is small and fast on CPU, so it needs no GPU (torch is installed as a CPU-only wheel).

OpenAI embeddings are available through the same LangChain `Embeddings` interface when speed or quality matters more than keeping data local:

```python
from conversation_cluster_memory import ConversationClusterMemory, openai_embeddings

memory = ConversationClusterMemory("conversation-42", embeddings=openai_embeddings())
```

The store records the embedding model of every vector. Opening a conversation, or querying its vectors, with another model raises `EmbeddingModelMismatchError`: two vector spaces are never mixed.

With this model, turns on the same topic often score only 0.5–0.7 cosine similarity. The default `similarity_threshold` of 0.8 may therefore produce many one-turn clusters; the evaluation makes this visible.

## Usage

```python
from conversation_cluster_memory import ConversationClusterMemory, ConversationKey

memory = ConversationClusterMemory(
    ConversationKey("conversation-42", user="user-7", tenant="tenant-1"),
    path="./memory",  # leave out to keep everything in memory
)
memory.add_message({"role": "system", "content": "You are a helpful assistant."})
memory.add_message({"role": "user", "content": "Hello"})
messages = memory.window()  # system messages, then the last `keep_recent_turns` turns
```

- **Conversation key.** `ConversationKey(conversation, user=None, tenant=None)`, a plain string (a key with only a conversation), or any object with a `key_parts()` method that returns an ordered tuple of strings. Each conversation has its own SQLite rows and its own Chroma collection.
- **Turns.** A turn is one user message plus everything until the next user message. `keep_recent_turns` (default 10) counts turns. A turn is embedded once, when it leaves the recent window; turns whose embedding failed are backfilled on the next `add_message`.
- **System messages.** A system message (such as the system prompt) belongs to no turn: it is never counted, embedded or summarized, and `window()` always returns it first.
- **Embedded text.** By default the user message, the assistant answer and, per tool result, only a header with the tool name and size. Pass `embedding_text=` to replace it. Tool results are always stored in full.
- **Storage.** With `path`, messages go to `archive.sqlite3` and vectors to `chroma/` in that directory. Without it, nothing touches disk.

The demo (`uv run python -m conversation_cluster_memory`) shows a built-in conversation and which turns have a vector.

## Evaluation results

<!-- eval-results:start -->
_No evaluation run yet. This block is replaced by the evaluation script after the first run._
<!-- eval-results:end -->
