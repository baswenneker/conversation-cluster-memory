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

> **Placeholder.** A short explainer video will appear here. It will be recorded over the [motion graph](docs/motion-graph/index.html) (`docs/motion-graph/index.html`), a click-through page that shows embedding, clustering, summarization and retrieval step by step. Open it locally in a browser to walk through the concept yourself: it needs no server and no network. Next, Back, the arrow keys and space step through it; `H` hides the controls.

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
messages = memory.window()  # system messages, the memory message, then the recent turns
```

- **Conversation key.** `ConversationKey(conversation, user=None, tenant=None)`, a plain string (a key with only a conversation), or any object with a `key_parts()` method that returns an ordered tuple of strings. Each conversation has its own SQLite rows and its own Chroma collection.
- **Turns.** A turn is one user message plus everything until the next user message. `keep_recent_turns` (default 10) counts turns. A turn is embedded once, when it leaves the recent window; turns whose embedding failed are backfilled on the next `add_message`.
- **System messages.** A system message (such as the system prompt) belongs to no turn: it is never counted, embedded or summarized, and `window()` always returns it first.
- **Embedded text.** By default the user message, the assistant answer and, per tool result, only a header with the tool name and size. Pass `embedding_text=` to replace it. Tool results are always stored in full.
- **Topic clusters.** Every `cluster_every_n_turns` turns (default 5), once at least `min_batch_turns` (default 8) embedded turns outside the window are unclustered, that batch is clustered by topic: agglomerative clustering (average linkage) on cosine similarity, with a small bonus for turns close in time, cut at `similarity_threshold` (default 0.8, a minimum similarity). A batch cluster whose centroid reaches the threshold against an existing cluster joins the most similar one (on a tie, the oldest); otherwise it becomes a new cluster. A clustered turn is never re-assigned. `memory.clusters()` lists them; pass `clustering=` to use another strategy.
- **Summaries.** Each new or grown cluster is summarized by a LangChain chat model: `summary_model=` takes a model name or any `BaseChatModel`; the default is `gpt-6.1-luna` (OpenAI), or `OPENAI_SUMMARY_MODEL` when set. The summary is structured output: text plus anchors (subject, period, names); the prompt keeps names, identifiers, numbers, dates and decisions literally and invents nothing. `resummarize_strategy="incremental"` (default) summarizes the old summary plus the new turns; `"full"` summarizes all original turns again. The archive keeps every original turn, linked to the summary that covers it. If the chat model fails, the cluster stays unsummarized, the error is logged and reported, and the next round retries. A summary that keeps more than 80% of its originals' tokens is accepted and flagged as `poor_compression` in the report.
- **The window.** `window()` returns the system messages, one system message "Conversation memory" with the cluster summaries in time order, then the recent turns verbatim. Turns outside the recent window appear only through their cluster's summary. In every turn but the current one, a tool result above `max_tool_result_tokens` (default 2000, counted with `tiktoken`) is replaced by its header; the archive keeps it in full. `window()` always reads the last committed state: a summary and the turns it covers are committed together.
- **Compaction rounds.** A round embeds the turns that left the window, clusters a batch when a check is due, and summarizes every new or grown cluster. `auto_compact="background"` (default): `add_message` starts the round in a background thread and returns at once. `"inline"`: `add_message` returns after the round. `"off"`: no round runs until you call `compact()` or `await acompact()`. One round runs at a time per memory; messages added during a round start another round right after it. `wait_idle()` waits until no round is running. A round interrupted before it commits (the process stops) leaves the previous state, and it is redone on the next `add_message`.
- **Reports.** Every round produces a `CompactionReport`: the turns it clustered, the summaries it made (with tokens of the originals and of the summary), errors, the window's tokens before and after, and the summary model's input and output tokens as the API reported them. It is passed to `on_compact=` and logged at INFO by the `conversation_cluster_memory` logger. "Before" is the window at the start of the round with the turns the round summarized still verbatim.
- **Storage.** With `path`, messages go to `archive.sqlite3` and vectors to `chroma/` in that directory. Without it, nothing touches disk.

The demo (`uv run python -m conversation_cluster_memory`) shows a built-in conversation, which turns have a vector, which turns fall into which cluster with their similarity to its centroid, each cluster's summary, and the window in which the summaries replace their turns. Summaries need `OPENAI_API_KEY`; without it the demo says so and shows the clusters unsummarized.

## Evaluation results

<!-- eval-results:start -->
_No evaluation run yet. This block is replaced by the evaluation script after the first run._
<!-- eval-results:end -->
