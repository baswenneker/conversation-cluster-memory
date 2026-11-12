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

> **Placeholder.** A short explainer video will appear here. It will be recorded over the [motion graph](docs/motion-graph/index.html) (`docs/motion-graph/index.html`), a click-through page that shows embedding, clustering, summarization and retrieval step by step. Open that page locally in a browser to walk through the concept yourself.

## Installation

[uv](https://docs.astral.sh/uv/) is the only supported install path; there is no pip requirements file. The project needs Python 3.13, which uv installs for you.

```bash
uv sync                                       # create .venv and install all dependencies
uv run pytest                                 # run the test suite
uv run python -m conversation_cluster_memory  # run the demo on a built-in conversation, in memory
```

The demo entry point and `.env.example` arrive with the first feature. Tests need no API key and no network. The default embedding model is downloaded once, the first time it is used. OpenAI components (OpenAI embeddings, the summary model, the evaluation judge) read `OPENAI_API_KEY` from a `.env` file; copy `.env.example` to `.env` and fill it in. Without a key, using such a component raises a clear error.

## Default embedding model

The default embedding model is [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), a local sentence-transformer. It was chosen because:

- it runs locally, so conversation text never leaves your machine to be embedded;
- it is multilingual, so conversations in languages other than English work without changing the model;
- it is small and fast on CPU, so it needs no GPU (torch is installed as a CPU-only wheel).

OpenAI embeddings are available through the same interface when speed or quality matters more than keeping data local.

With this model, turns on the same topic often score only 0.5–0.7 cosine similarity. The default `similarity_threshold` of 0.8 may therefore produce many one-turn clusters; the evaluation makes this visible.

## Usage

> Each feature adds its usage example here as it is built.

## Evaluation results

<!-- eval-results:start -->
_No evaluation run yet. This block is replaced by the evaluation script after the first run._
<!-- eval-results:end -->
