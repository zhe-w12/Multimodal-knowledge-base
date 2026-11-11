"""Keep pytest runs deterministic even when a local .env enables cloud services."""

import os


os.environ["RAG_LLM_PROVIDER"] = "rule"
os.environ["RAG_EMBEDDING_PROVIDER"] = "hash"
os.environ["RAG_RERANK_PROVIDER"] = "rule"
os.environ["RAG_VECTOR_STORE_PROVIDER"] = "memory"
os.environ["RAG_EMBEDDING_DIMENSIONS"] = "64"
# Legacy API tests exercise application behavior without account setup; the
# dedicated authorization tests turn this switch on explicitly.
os.environ["RAG_AUTH_ENFORCEMENT_ENABLED"] = "0"
