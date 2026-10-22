"""Dependency-free frozen metadata for supported memory backends."""

from __future__ import annotations


MEM0_OPENAI_EMBEDDING_PROVIDER = "openai"
MEM0_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
MEM0_OPENAI_EMBEDDING_DIMENSIONS = 1536
MEM0_VERSION = "2.0.2"
MEM0_ENTITY_MODEL = "en_core_web_sm@3.8.0"
MEM0_SPACY_VERSION = "3.8.16"
MEM0_SUBMODULE_COMMIT = "70bc9e51d57fe005d02b7b6d81b56476bade3cb3"
MEM0_QDRANT_DISTANCE = "Cosine"
MEM0_QDRANT_ON_DISK = True
MEM0_QDRANT_BM25_ENABLED = False
MEM0_RETRIEVAL_SIMILARITY_THRESHOLD = 0.1
MEM0_RETRIEVAL_RERANK = False
MEM0_HISTORY_DB_RELATIVE_PATH = "backend_state/mem0_history.sqlite3"

#: How the `infer=True` path builds the inputs mem0 extracts from. It decides
#: what mem0 infers over, so it belongs to the frozen run configuration
#: (`mem0_embedding_chunk_policy()`, `Mem0RuntimeConfig`): two runs on
#: different strategies are different runs.
#:
#: `legacy_no_message_split` is what every plan frozen before 2026-09-18 ran:
#: a pool is batched at message boundaries, and a message that alone exceeds
#: the embedding budget is a hard error rather than anything to cut.
#: `message_split_v1` cuts such a message into consecutive, non-overlapping
#: sub-messages first, so the turn can be written at all.
MEM0_EXTRACTION_CHUNK_STRATEGY_LEGACY = "legacy_no_message_split"
MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1 = "message_split_v1"

#: New plans split oversized messages. Resumed runs explicitly pass the
#: strategy recorded in their own frozen configuration to the adapter.
MEM0_EXTRACTION_CHUNK_STRATEGY = MEM0_EXTRACTION_CHUNK_STRATEGY_MESSAGE_SPLIT_V1

MEMOS_VERSION = "2.0.15"
MEMOS_SUBMODULE_COMMIT = "9a98793dd9486bc384f73ecb7e18cd6c9a01d7d9"
MEMOS_SOURCE_SHA256 = "56fc72af3fa2f5045b7b32179c16d7a465ce8ed8f1506ae828b49a331ed642e8"
MEMOS_MEMPA_PATCH_VERSION = "20260914.1"
