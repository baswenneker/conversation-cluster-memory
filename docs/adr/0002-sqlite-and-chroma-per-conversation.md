# SQLite for records, Chroma for vectors, one collection per conversation

Messages, turns, clusters, summaries and counters live in SQLite; vectors live in Chroma, with one collection per conversation. Both sit behind interfaces so a production store can replace them.

SQLite holds relations, counters and atomic commits that a vector store handles poorly, and gives keyword search (FTS5) for free. A collection per conversation makes a leak between tenants structurally impossible instead of depending on a filter in every query.

## Considered Options

- Everything in Chroma, messages as metadata: one system, but counters, relations and transactions fit badly.
- One shared collection with a conversation filter: simpler, but one forgotten filter exposes another tenant's archive.

## Consequences

- Two stores must stay consistent: a round commits in SQLite first and writes vectors after; a missing vector is backfilled by the next round.
- Thousands of conversations mean thousands of collections.
