# Cluster whole turns in batches, not single messages online

The first plan assigned each message, as soon as it was embedded, to the nearest cluster centroid (online, per message). We cluster whole turns instead, in batches, once enough turns have left the recent window: agglomerative clustering on cosine similarity with a bonus for turns close in time, after which each batch cluster may join an existing cluster.

Per-message clustering split a short question from its long answer or tool result, because their vectors rarely reach the threshold together; a summary then covered an answer without its question. Online assignment also clustered every message in isolation, without the context of its neighbours. Batches with a proximity bonus keep a topic's consecutive turns together, and the turn is the unit the window already works in.

## Considered Options

- Online per message (the first plan): simplest, but splits questions from answers.
- Louvain community detection, as in an earlier podcast segmenter: finds topics without a cluster count, but is non-deterministic and built for large graphs, not batches of about eight turns.

## Consequences

- Retrieval works per cluster, not per message: an exact figure is found through its cluster's summary or original turns. Per-message vectors can be added later.
- A turn is clustered only after `min_batch_turns` turns have piled up outside the window, so the newest archive is briefly unclustered.
