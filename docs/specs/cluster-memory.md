# Conversation cluster memory

A memory layer that keeps a conversation's model context bounded — for conversations that last months or a whole subscription — without making earlier facts unreachable.

## Problem Statement

An application that talks to a chat model sends the conversation so far with every turn. A long-running conversation grows until it no longer fits the model's context window, and long before that it becomes slow and expensive: every turn re-sends everything. Large tool results (lists, long series of records) make this much worse. Cutting old messages off keeps the window small but makes the assistant forget facts it was told weeks ago. The application needs a layer that decides *which* text the model still sees: recent turns verbatim, older history compressed by topic, and the rest searchable on demand.

## Solution

A Python library, `ConversationClusterMemory`, that sits beside whatever inference stack the application already uses. The application keeps calling its own model; the library owns the memory:

- Every message is stored, per conversation, in a persistent archive. Nothing is ever deleted from the archive by compression.
- The last `keep_recent_turns` turns stay verbatim in the window.
- Turns that leave the recent window are embedded (one vector per turn), and once enough of them have piled up they are clustered by topic in a batch. A new batch cluster that resembles an existing cluster joins it.
- Each cluster is summarized by a chat model. In the window, the summary replaces the cluster's turns; in the archive, the originals remain.
- On a new question, retrieval finds relevant old clusters (vector search plus keyword search) and adds them to the window within a fixed token budget. The model can also search memory itself through a tool.
- When cluster summaries themselves grow too large, the oldest are summarized again into a period summary.
- An automatic evaluation (DeepEval) measures, on fixed synthetic conversations, how many tokens each compaction saves and whether answers to the same questions stay correct after compression.

The library is domain-neutral: nothing in it, its docs, examples or test data refers to a specific client, product or industry.

All parameters are configurable through the constructor; the defaults below are starting points for evaluation, not frozen decisions.

| Parameter | Default | Meaning |
|---|---|---|
| `keep_recent_turns` | 10 | Turns kept verbatim in the window. A turn is embedded when it leaves this window. |
| `min_batch_turns` | 8 | Unclustered turns outside the window needed before a batch is clustered. |
| `cluster_every_n_turns` | 5 | How often (in turns) the compaction check runs. |
| `similarity_threshold` | 0.8 | Minimum cosine **similarity** (not distance) to join a cluster. |
| `resummarize_strategy` | `incremental` | `incremental`: old summary + new turns. `full`: all original turns. |
| `max_tool_result_tokens` | 2000 | Tool results above this are shortened to their header in earlier turns of the window. |
| `auto_compact` | `background` | `background`, `inline` or `off`. |
| `auto_retrieval` | on | Add retrieval hits for the latest user message to the window. |
| `retrieval_k` | 3 | Maximum retrieval hits. |
| `retrieval_token_budget` | 1500 | Maximum tokens retrieval may add to the window. |
| `retrieval_min_similarity` | 0.5 | Minimum cosine similarity for a vector hit. |
| `max_summary_tokens` | 3000 | When summaries in the window exceed this, the oldest are merged into a level-2 summary. |

## User Stories

1. As an application developer, I want to add messages to a memory object and get back the window to send to my model, so that I never manage context size myself.
2. As an application developer, I want the memory to be bound to a conversation key (conversation, optionally user and tenant, or my own key type), so that one tenant's history never appears in another tenant's window or search results.
3. As an application developer, I want the memory to persist to a directory and to run fully in memory when I give no path, so that I can use it in production and in a demo or test without a server.
4. As an application developer, I want a local multilingual embedding model by default and OpenAI embeddings as an option, behind one interface, so that data can stay on my machine or I can trade that for speed.
5. As an application developer, I want the memory to refuse a query or a write with an embedding model other than the one that built the archive, so that two vector spaces are never silently mixed.
6. As an application developer, I want large tool results stored in full but embedded and shown in the window as a short header, so that one large result does not blow up the window or get silently truncated by the embedding model.
7. As an application developer, I want older turns grouped into topic clusters, so that a summary covers one coherent topic and not an arbitrary stretch of time.
8. As an application developer, I want turns that come back to an earlier topic to join that topic's cluster, so that the topic keeps one summary instead of many fragments.
9. As an application developer, I want compaction to run in the background after `add_message` by default, so that my user never waits for a summarization call.
10. As an application developer, I want a report per compaction round (what was clustered and summarized, tokens before and after, cost), through a callback and the log, so that I can monitor savings in production.
11. As an application developer, I want a failed summarization to leave the cluster unsummarized and be retried next round, so that a model outage never loses history or corrupts the window.
12. As an application developer, I want `window()` to always return the last complete state, even while a round is running, so that a half-summarized cluster never reaches the model.
13. As an end user, I want to ask "do you remember when we discussed X?" and get an answer about X, so that a months-long conversation still feels continuous.
14. As an application developer, I want a `recall_memory` tool I can give my model, so that the model can search its own memory with a query it writes itself.
15. As an application developer, I want retrieval to stay within a token budget and never duplicate what is already in the window, so that retrieval does not grow the window back.
16. As a maintainer, I want an automatic evaluation that replays fixed conversations and asks the same questions before and after every compaction round, so that a change that drops facts fails visibly.
17. As a maintainer, I want token savings per round measured in the same evaluation run as answer quality, so that a saving that costs correctness is not called a success.
18. As a maintainer, I want the README to show the latest evaluation results, updated by a script after each run, so that the numbers are never hand-edited or stale.
19. As a maintainer, I want a local sweep command that compares parameter values, so that the defaults are chosen from numbers, not gut feeling.
20. As an application developer running a conversation for a year, I want old summaries merged into period summaries while the originals stay in the archive, so that the window stays bounded forever.
21. As an application developer, I want an explicit re-index when I change embedding model, so that switching models is a controlled step.
22. As an application developer, I want to delete everything for a conversation, a user or a tenant, so that I can honour a deletion request or the end of a subscription.
23. As a product owner, I want a cost estimate for a year of conversation from measured per-round costs, so that I can price the feature.
24. As a presenter, I want a standalone, click-through animated page that explains embedding, clustering, summarization and retrieval, so that I can record an explainer video with my camera as an overlay.
25. As a reader of the repository, I want an English plan document with a framework-agnostic specification I can paste into another coding agent, so that the concept can be rebuilt in another stack (for example the Vercel AI SDK).

## Implementation Decisions

**Public interface.** One class, `ConversationClusterMemory`, is the public API: `add_message()`, `window()`, `compact()` / `acompact()`, `wait_idle()`, `retrieve()`, `as_tool()`, `reindex()`, and the delete operations. A `ConversationKey(conversation, user=None, tenant=None)` identifies a conversation; only `conversation` is required, and a plain string is accepted as shorthand for a key with only a conversation. An application with its own key type supplies any object that yields an ordered tuple of strings (a small protocol); the memory derives the storage identity and the collection name from that tuple. The message store, the clustering, the window manager and retrieval are internal modules with their own unit tests.

**Turn.** A turn is one user message plus everything that follows until the next user message: assistant messages, tool calls and tool results. All thresholds count turns, never messages, so a turn is never split.

**Storage.** Messages, turns, clusters, summaries and counters live in SQLite; vectors live in Chroma. Each sits behind its own interface so a production store (for example PGVector) can replace it. Without a path both run in memory (SQLite in memory, Chroma ephemeral client). Each conversation has its own Chroma collection, so a missing filter can never leak another tenant's data. The store records which embedding model produced each vector.

**Embedding.** Default: a local sentence-transformer, `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, fast on CPU). OpenAI embeddings are an option through the same LangChain `Embeddings` interface. Torch is installed as a CPU-only wheel. The unit of embedding is the turn. The text embedded is produced by a replaceable function `embedding_text`; its default is the user message, the assistant answer, and per tool result only a header (tool name and size). An option embeds a short LLM summary of the turn instead. The README names the default model and why it was chosen. The OpenAI API key comes from `OPENAI_API_KEY`, loaded from a `.env` file; the repository ships `.env.example`. Missing key when an OpenAI component is used: a clear error, no fallback.

**Compaction round.** Every `cluster_every_n_turns` turns, if at least `min_batch_turns` unclustered turns lie outside the recent window, the round:
1. embeds the turns that left the window and have no vector yet (backfill included);
2. clusters the batch with agglomerative clustering (average linkage, cut at `1 - similarity_threshold`) on cosine similarity plus a proximity bonus for turns that are close in time (as in the earlier podcast segmenter: bonus divided by the distance in turns). The clustering is deterministic and sits behind an interface so another strategy can be added;
3. compares each batch cluster's centroid with existing clusters; at or above `similarity_threshold` it joins the most similar one (ties: the oldest), else it becomes a new cluster;
4. summarizes each new or grown cluster with the configured chat model, following `resummarize_strategy`. A batch cluster of a single turn is still summarized, so no turn waits forever.

The summary is structured output: text plus anchors (subject, period, names). The prompt keeps names, identifiers, numbers, dates and decisions literally and invents nothing. Default summary model: `gpt-6.1-luna`, overridable by constructor or `OPENAI_SUMMARY_MODEL`.

**Commit and failure.** A round commits in one SQLite transaction (summary and cluster state together); the summary's vector is written to Chroma afterwards, and a summary without a vector is embedded by the next round. If the chat model fails, the cluster stays unsummarized, the turns stay in the batch, the error is logged and reported, and the next round retries. A summary barely shorter than its originals is accepted and flagged as poor compression in the report.

**Background compaction.** `auto_compact="background"` (default): `add_message` checks for work and, if any, starts a round as a background task (in the running event loop, else a thread) and returns at once. `"inline"`: `add_message` waits for the round (used by tests and evaluation). `"off"`: the application calls `compact()` itself. One round at a time per conversation (a lock); work that arrives during a round starts another round right after. `wait_idle()` waits for running rounds. Each round produces a report (clustered turns, summaries, tokens before and after, cost from API `usage`) passed to an `on_compact` callback and the log.

**Window.** `window()` returns: one system message "Conversation memory" holding the cluster summaries in time order (by oldest member), then retrieval hits, then the last `keep_recent_turns` turns verbatim. Tool results above `max_tool_result_tokens` in earlier turns are replaced by their header; the current turn keeps full tool results. Token counts in the window use `tiktoken` with the tokenizer of the configured model.

**Retrieval.** With `auto_retrieval` on, `window()` uses the latest user message as query. Search is hybrid: a vector search on each cluster's summary vector (or its centroid when not yet summarized) with `retrieval_min_similarity`, plus a keyword search (BM25 over summaries and anchors, SQLite FTS5) with its own minimum score. An optional query rewrite lets the chat model turn the message into a search query first. Hits are sorted by score, deduplicated against the window, and added until `retrieval_token_budget` is reached (at most `retrieval_k`). A hit's payload is the summary if present, else the original turns in time order. Search never leaves the conversation's own collection. `retrieve(query, k)` is public, and `as_tool()` returns a LangChain tool `recall_memory(query)`. No per-message vectors for now; they are added later if evaluation shows exact figures being lost.

**Lifetime.** When summaries in the window exceed `max_summary_tokens`, the oldest summaries in time are merged into a level-2 (period) summary; level-1 summaries stay in the archive and retrieval searches both levels. Opening an archive with a different embedding model raises an error that points to `reindex(new_embeddings)`, which re-embeds the archive explicitly. `delete_conversation`, `delete_user` and `delete_tenant` remove SQLite rows and Chroma collections; `delete_user` and `delete_tenant` only match keys that set those fields. Time-based retention is the application's job. A cost command extrapolates a year of use from measured per-round costs and a turns-per-week figure, without new model calls.

**Evaluation.** DeepEval replays three synthetic, domain-neutral conversations, generated once by a script with planted facts and then committed: short (30 turns, one topic), long (200 turns with large tool results), switching (80 turns returning to earlier topics); about 15 questions each with gold answers. With `auto_compact="inline"`, after every compaction round the same questions are asked in two conditions: the uncompressed window, and the compressed window plus retrieval. Metrics: answer relevancy, contextual precision, contextual recall, faithfulness, and a GEval factual-correctness metric against the gold answer. A custom metric reports tokens before and after each round, `(before − after) / before`, and the cost of embedding, summarizing and judging. A run fails when the compressed score falls below 90% (configurable) of the uncompressed score, or when the token measurement is missing. Judge model: `gpt-6.1-terra`, overridable by `OPENAI_JUDGE_MODEL`. A script replaces a marked block in the README with the latest run (date in UTC, models per role, scores per condition, token savings, cost); older runs stay as files. CI runs the evaluation on `main` and by hand, never on pull requests, with `OPENAI_API_KEY` as repository secret, and commits the README block to `main` only when the numbers changed. A local sweep command compares parameter values and writes a table. DeepEval is not part of the factory gate.

**Repository.** Python 3.13, uv only (no pip requirements file), src layout. Everything in the repository is English. A demo entry point grows with each ticket and shows the store, the clusters, a summary and retrieval on a built-in conversation, in memory.

**Motion graph.** A standalone HTML/CSS/JavaScript page, no build step, opened locally and full-screen, styled with the HeadingFWD Terminal design system (dark only, JetBrains Mono only, one cyan accent, hairlines, tight display tracking, wide label tracking). Eight steps, one per click: messages form turns; the window of recent turns; turns leaving the window become vectors; a batch is clustered with cosine and a proximity bonus; a batch cluster joins an existing cluster; the cluster is summarized and a token counter drops; the window becomes the memory message plus recent turns; a question triggers hybrid retrieval and hits enter the window within budget. Each step's animation plays once and stops; Next, Back, arrow keys and space navigate; no autoplay, no timers, no loops. The bottom-right quarter stays empty for a camera overlay; the `H` key hides the controls.

**Documentation.** An English plan document in the repository holds the complete plan with all decisions above and, as one whole, the framework-agnostic specification (below). The README covers installation with uv, the default embedding model and why, usage, a placeholder for the explainer video (a recording by the owner, never fabricated), and the marked evaluation-results block with a placeholder until the first run.

## Testing Decisions

Good tests exercise external behaviour through `ConversationClusterMemory` (and, for internal modules, their own small interfaces), with fake embeddings and a fake chat model from LangChain so the gate needs no network, no model download and no API key. Deterministic vectors let tests set exact cosine similarities.

- Message store: thresholds in turns, embedding exactly when a turn leaves the recent window, backfill, model-id mismatch refused, isolation between conversation keys, persistence across reopen, in-memory mode.
- Clustering: a batch is clustered only when `min_batch_turns` is reached and only every `cluster_every_n_turns`; similar turns share a cluster, a topic change opens a new one; the threshold shifts the outcome; a returning topic joins the existing cluster; ties go to the oldest cluster; same input, same clusters.
- Window manager: summary replaces turns in the window but not in the archive; both re-summarize strategies; failure leaves the cluster unsummarized and retries; large tool results shortened only in earlier turns; background, inline and off modes; one round at a time; `window()` during a round returns the last committed state; restart after an interrupted round.
- Retrieval: vector and keyword hits, minimum similarity, budget and `k`, deduplication, summary versus originals as payload, no cross-conversation hits, the tool.
- Lifetime: level-2 summaries above `max_summary_tokens`, reindex, the three deletes.
- Evaluation: the token metric and the README updater are unit-tested; the DeepEval run itself runs in its own CI workflow.
- Motion graph: a headless browser test that steps through all eight steps, checks that nothing advances without input, and that the camera area stays empty.

## Out of Scope

- Moving any application's turn loop to LangChain.
- Searching across conversations of the same tenant.
- Re-assigning already clustered turns (periodic re-clustering).
- Per-message vectors and a second, stricter retrieval threshold.
- Time-based retention policies.
- A server, API or UI around the library (beyond the demo and the motion graph).
- Recording the explainer video.
- Any client- or industry-specific example, data or integration.

## Further Notes

- With the multilingual MiniLM model, sentences on the same topic often score 0.5–0.7 cosine similarity, so the default `similarity_threshold` of 0.8 may produce many one-turn clusters. The evaluation must make this visible and the README says so.
- The batch clustering borrows two ideas from an earlier podcast segmenter: embedding a short summary instead of raw text (here an option) and a proximity bonus for neighbours. Its Louvain community detection is not used: it is non-deterministic and built for large graphs.

### Framework-agnostic specification (copy-ready)

**Purpose.** Keep a conversation's model context bounded — including conversations that last months or a full subscription — without making earlier facts unreachable. A memory layer sits beside whatever inference stack the app already uses. The app calls the model; this layer decides *which* text the model sees: it stores messages, embeds turns when they leave the recent window, clusters them by cosine similarity in batches, summarizes clusters, and retrieves relevant archive for a new question. It does not require any particular framework (LangChain, the Vercel AI SDK or another); those are interchangeable hosts.

**Building blocks.** A message store (persistent, with optional vectors), a clustering step, a window manager, retrieval, and evaluation. Around them three adapters: an embedding model (text to fixed-dimension vector), a chat model (prompt to completion, for summarizing and judging), and a vector index (save vectors, return nearest neighbours by cosine similarity). Swap the adapters and the memory logic stays the same.

**Isolation.** Every store is bound to a conversation key (a conversation id, optionally scoped by user and tenant). Vectors, clusters and summaries of one tenant never appear in another tenant's window or retrieval. Prefer one index or collection per conversation over a shared index with a filter.

**Turns.** A turn is one user message plus everything until the next user message (assistant messages, tool calls, tool results). All thresholds count turns.

**Store and embedding.** Every message is stored with role, content, timestamp, turn number and a stable id. Originals are never deleted by compression; they are the archive. The last `keep_recent_turns` (default 10) turns stay verbatim in the window and are not embedded. A turn is embedded once, when it leaves that window; missing vectors are backfilled. The embedded text is the user message, the answer, and a header per tool result (optionally a short model-written summary of the turn). The store records which embedding model made each vector; a different model requires an explicit re-index, never a silent mix. Large tool results stay in the store in full.

**Clustering.** Cosine similarity, threshold `similarity_threshold` (default 0.8) as minimum **similarity**, not distance. Every `cluster_every_n_turns` (default 5) turns, if at least `min_batch_turns` (default 8) unclustered turns lie outside the window, cluster that batch deterministically (for example agglomerative, average linkage, cut at `1 − threshold`), with a bonus for turns close in time. Each batch cluster whose centroid reaches the threshold against an existing cluster joins it (ties: oldest); otherwise it is a new cluster. Clustered turns are not re-assigned later.

**Summarizing and the window.** Each new or grown cluster is summarized by the chat model: either the old summary plus the new turns (incremental, default) or all original turns (full). The summary keeps names, identifiers, numbers, dates and decisions literally, invents nothing, and carries short anchors (subject, period, names). It replaces the cluster's turns in the window only; the archive keeps the originals, marked summarized. The window is: one memory message with the summaries in time order, retrieval hits, and the recent turns verbatim; tool results above a size limit in earlier turns are replaced by their header. A failed summary leaves the cluster unsummarized and is retried next round. Rounds run in the background, one at a time per conversation, and commit atomically; a reader always sees the last complete state.

**Retrieval.** Embed the query with the same model as the store (refuse otherwise). Search this conversation only: by vector against each cluster's summary vector (or centroid when unsummarized), with a minimum similarity, and by keywords (BM25) over summaries and anchors. Optionally let the chat model rewrite the query first, or expose retrieval as a tool the model calls. Payload: the summary if present, else the original turns. Sort, deduplicate against the window, and stop at a fixed token budget.

**Lifetime.** The window is always a window. When summaries exceed a token limit, merge the oldest into a period summary; earlier levels stay in the archive and retrieval searches all levels. Persistent storage lets any session resume. A new embedding model means an explicit re-index. Deleting a conversation, user or tenant removes its store and vectors.

**Evaluation.** Replay fixed conversations with planted facts. After every compaction round, ask the same questions with the uncompressed window and with the compressed window plus retrieval. Measure answer relevancy, contextual precision and recall, faithfulness, and factual correctness against gold answers; in the same run, count window tokens before and after each round and report `(before − after) / before` plus the cost of embedding, summarizing and judging. Fail when compressed quality drops below a configurable share of uncompressed quality. Show the latest results in the README, updated by a script.

**Not frozen.** Production values of the thresholds, the embedding model, chat model and vector index of a deployment, cross-conversation retrieval, and the quality acceptance level.
