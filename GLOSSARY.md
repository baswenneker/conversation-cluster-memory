# Glossary

**Conversation key**: The identity of one conversation: a conversation id, optionally scoped by user and tenant. Everything in memory belongs to exactly one key and is never visible under another.
_Avoid_: session, thread id

**Turn**: One user message plus everything that follows until the next user message: assistant messages, tool calls and tool results. All thresholds count turns.
_Avoid_: exchange, round

**Recent window**: The last `keep_recent_turns` turns, sent to the model verbatim and not yet embedded.
_Avoid_: buffer, short-term memory

**Archive**: Every message ever stored for a conversation, with its vectors and summaries. Compression never removes anything from it.
_Avoid_: history, backlog

**Batch**: The unclustered turns outside the recent window that a compaction round clusters together.

**Cluster**: A set of turns about one topic, with a centroid and, once summarized, a summary.
_Avoid_: segment, topic group

**Centroid**: The normalized mean of the turn vectors of a cluster.

**Summary vector**: The embedding of a cluster's summary text. Retrieval prefers it over the centroid.

**Compaction round**: One run that embeds turns leaving the window, clusters a batch, attaches clusters to existing ones and summarizes them.
_Avoid_: compression job, cycle

**Window**: What the model receives: the memory message with summaries, retrieval hits, and the recent window.
_Avoid_: prompt, context (on its own)

**Retrieval hit**: A cluster found for a query, delivered as its summary or, if not yet summarized, its original turns.

**Level-2 summary**: A summary of the oldest summaries, made when summaries in the window exceed their token limit.
_Avoid_: meta-summary
