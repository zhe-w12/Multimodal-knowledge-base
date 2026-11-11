"""Zilliz Cloud / Milvus vector-store adapter."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from manufacturing_rag.adapters.embedding import EmbeddingAdapter, build_embedding_adapter
from manufacturing_rag.config import DEFAULT_CONFIG, RAGConfig
from manufacturing_rag.query import split_terms
from manufacturing_rag.adapters.retriever import rrf_fuse_results
from manufacturing_rag.schemas import DEFAULT_WORKSPACE_ID, DocumentChunk, EntityExtractionResult, RetrievalResult


@dataclass(frozen=True)
class VectorStoreStatus:
    provider: str
    connected: bool
    endpoint_configured: bool
    token_configured: bool
    collection_name: str
    collection_exists: bool = False
    row_count: Optional[int] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "connected": self.connected,
            "endpoint_configured": self.endpoint_configured,
            "token_configured": self.token_configured,
            "collection_name": self.collection_name,
            "collection_exists": self.collection_exists,
            "row_count": self.row_count,
            "message": self.message,
        }


def hybrid_candidate_limit(result_limit: int) -> int:
    """Over-fetch vector candidates so lexical reranking can recover exact page matches."""

    safe_limit = max(1, int(result_limit))
    return min(max(safe_limit * 4, 20), 100)


class MilvusVectorStore:
    """Adapter for the local Docker Milvus service."""

    vector_field = "vector"
    sparse_vector_field = "sparse_vector"

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG, embedding: Optional[EmbeddingAdapter] = None) -> None:
        self.config = config
        self.embedding = embedding or build_embedding_adapter(config, check_dimensions=True)
        self.endpoint = os.getenv(config.zilliz_endpoint_env, "").strip()
        self.token = os.getenv(config.zilliz_token_env, "").strip()
        self.collection_name = config.zilliz_collection_name
        self._connected = False

    @property
    def provider_name(self) -> str:
        """Expose the configured remote provider while keeping direct adapter compatibility."""

        return self.config.vector_store_provider if self.config.vector_store_provider in {"milvus", "zilliz"} else "zilliz"

    @property
    def is_local_endpoint(self) -> bool:
        """Allow only loopback or the Docker Compose Milvus service hostname."""

        endpoint = self.endpoint if "://" in self.endpoint else f"http://{self.endpoint}"
        hostname = (urlparse(endpoint).hostname or "").lower()
        return hostname in {"localhost", "127.0.0.1", "::1", "milvus"}

    def status(self) -> VectorStoreStatus:
        if not self.endpoint or not self.token:
            return VectorStoreStatus(
                provider=self.provider_name,
                connected=False,
                endpoint_configured=bool(self.endpoint),
                token_configured=bool(self.token),
                collection_name=self.collection_name,
                message="Milvus endpoint or token is not configured",
            )
        if not self.is_local_endpoint:
            return VectorStoreStatus(
                provider=self.provider_name,
                connected=False,
                endpoint_configured=True,
                token_configured=bool(self.token),
                collection_name=self.collection_name,
                message="Online Milvus endpoints are disabled; use the local Docker Milvus service",
            )
        try:
            utility, Collection = self._connect_and_import()
            exists = utility.has_collection(self.collection_name, using=self.config.zilliz_alias)
            row_count = None
            if exists:
                collection = Collection(self.collection_name, using=self.config.zilliz_alias)
                row_count = collection.num_entities
            return VectorStoreStatus(
                provider=self.provider_name,
                connected=True,
                endpoint_configured=True,
                token_configured=True,
                collection_name=self.collection_name,
                collection_exists=exists,
                row_count=row_count,
                message="connected",
            )
        except Exception as exc:  # noqa: BLE001 - status endpoint should return diagnostics without secrets.
            return VectorStoreStatus(
                provider=self.provider_name,
                connected=False,
                endpoint_configured=True,
                token_configured=True,
                collection_name=self.collection_name,
                message=str(exc),
            )

    def ensure_collection(self, drop_existing: bool = False) -> VectorStoreStatus:
        utility, Collection = self._connect_and_import()
        if drop_existing and utility.has_collection(self.collection_name, using=self.config.zilliz_alias):
            utility.drop_collection(self.collection_name, using=self.config.zilliz_alias)
        if not utility.has_collection(self.collection_name, using=self.config.zilliz_alias):
            self._create_collection()
        collection = Collection(self.collection_name, using=self.config.zilliz_alias)
        self._ensure_index(collection)
        collection.load()
        return VectorStoreStatus(
            provider=self.provider_name,
            connected=True,
            endpoint_configured=True,
            token_configured=True,
            collection_name=self.collection_name,
            collection_exists=True,
            row_count=collection.num_entities,
            message="collection ready",
        )

    def upsert_chunks(self, chunks: Iterable[DocumentChunk]) -> Dict[str, Any]:
        chunk_list = list(chunks)
        if not chunk_list:
            return {"status": "ok", "inserted": 0, "collection_name": self.collection_name}
        _, Collection = self._connect_and_import()
        self.ensure_collection(drop_existing=False)
        collection = Collection(self.collection_name, using=self.config.zilliz_alias)
        ids = [chunk.id for chunk in chunk_list]
        self._delete_existing(collection, ids)
        schema_fields = {getattr(field, "name", "") for field in getattr(collection.schema, "fields", [])}
        rows = self._chunks_to_rows(chunk_list, include_workspace_id="workspace_id" in schema_fields)
        mutation = collection.insert(rows)
        collection.flush()
        return {
            "status": "ok",
            "inserted": len(rows),
            "collection_name": self.collection_name,
            "mutation_result": str(mutation),
        }

    def replace_chunks(self, chunks: Iterable[DocumentChunk]) -> Dict[str, Any]:
        chunk_list = list(chunks)
        self.ensure_collection(drop_existing=True)
        if not chunk_list:
            return {"status": "ok", "inserted": 0, "collection_name": self.collection_name, "mode": "replace"}
        result = self.upsert_chunks(chunk_list)
        result["mode"] = "replace"
        return result

    def delete_chunk_ids(
        self,
        chunk_ids: Iterable[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        ids = list(dict.fromkeys(str(chunk_id)[:256] for chunk_id in chunk_ids if str(chunk_id)))
        total = len(ids)
        if not total:
            if progress_callback:
                progress_callback(0, 0)
            return {"status": "ok", "deleted": 0, "collection_name": self.collection_name}

        utility, Collection = self._connect_and_import()
        if not utility.has_collection(self.collection_name, using=self.config.zilliz_alias):
            if progress_callback:
                progress_callback(total, total)
            return {"status": "ok", "deleted": 0, "collection_name": self.collection_name}

        collection = Collection(self.collection_name, using=self.config.zilliz_alias)
        safe_batch_size = max(1, int(batch_size))
        for offset in range(0, total, safe_batch_size):
            batch = ids[offset : offset + safe_batch_size]
            self._delete_existing(collection, batch)
            if progress_callback:
                progress_callback(min(offset + len(batch), total), total)
        # Flush once after all delete requests to avoid repeating a synchronous
        # Milvus round trip for every batch.
        collection.flush()
        return {"status": "ok", "deleted": total, "collection_name": self.collection_name}

    def search(
        self,
        query: str,
        entities: EntityExtractionResult,
        limit: int = 10,
        workspace_ids: Optional[Iterable[str]] = None,
    ) -> List[RetrievalResult]:
        _, Collection = self._connect_and_import()
        self.ensure_collection(drop_existing=False)
        collection = Collection(self.collection_name, using=self.config.zilliz_alias)
        collection.load()
        schema_fields = {getattr(field, "name", "") for field in getattr(collection.schema, "fields", [])}
        filter_expression = None
        if workspace_ids is not None:
            allowed_workspace_ids = sorted({str(value) for value in workspace_ids if value})
            if not allowed_workspace_ids:
                return []
            if "workspace_id" not in schema_fields:
                return []
            values = ", ".join(json.dumps(value, ensure_ascii=False) for value in allowed_workspace_ids)
            filter_expression = f"workspace_id in [{values}]"
        vector = self.embedding.embed_text(query)
        query_terms = terms_for_vector_search(query, entities)
        search_params = {"metric_type": self.config.zilliz_metric_type, "params": {}}
        candidate_limit = hybrid_candidate_limit(limit)
        output_fields = ["text", "source", "page", "block_id", "modality", "metadata_json", "image_path"]
        if "workspace_id" in schema_fields:
            output_fields.append("workspace_id")
        sparse_vector = self._sparse_vector(query)
        dense_hits = collection.search(
            data=[vector],
            anns_field=self.vector_field,
            param=search_params,
            limit=candidate_limit,
            expr=filter_expression,
            output_fields=output_fields,
            consistency_level=self.config.zilliz_consistency_level,
        )
        route_results = [self._hits_to_results(dense_hits, query_terms, "milvus_dense")]
        if sparse_vector:
            try:
                sparse_hits = collection.search(
                    data=[sparse_vector],
                    anns_field=self.sparse_vector_field,
                    param={"metric_type": "IP", "params": {}},
                    limit=candidate_limit,
                    expr=filter_expression,
                    output_fields=output_fields,
                    consistency_level=self.config.zilliz_consistency_level,
                )
                route_results.append(self._hits_to_results(sparse_hits, query_terms, "milvus_sparse"))
            except Exception:
                # Older collections may not have a sparse field. Dense recall
                # remains useful, while local BM25/sparse fusion is handled by
                # MilvusHybridRetriever.
                pass
        return rrf_fuse_results(
            route_results,
            limit=limit,
            rrf_k=self.config.rrf_k,
            route_names=["milvus_dense", "milvus_sparse"],
        )

    def _hits_to_results(self, hits: Any, query_terms: List[str], channel: str) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        hit_rows = hits[0] if hits else []
        for rank, hit in enumerate(hit_rows, start=1):
            entity = hit.entity
            metadata = parse_metadata(entity.get("metadata_json"))
            workspace_id = entity.get("workspace_id")
            if workspace_id:
                metadata["workspace_id"] = str(workspace_id)
            chunk = DocumentChunk(
                id=str(hit.id),
                text=str(entity.get("text") or ""),
                source=str(entity.get("source") or "unknown"),
                page=none_if_negative(entity.get("page")),
                block_id=empty_to_none(entity.get("block_id")),
                modality=str(entity.get("modality") or "ocr_text"),
                metadata=metadata,
                image_path=empty_to_none(entity.get("image_path")),
                parent_id=empty_to_none(metadata.get("parent_id")),
                chunk_level=str(metadata.get("chunk_level") or "child"),
                confidence=float(metadata.get("confidence", 1.0)),
                needs_review=bool(metadata.get("needs_review", False)),
            )
            vector_score = float(hit.score)
            lexical_score = lexical_score_for_chunk(query_terms, chunk)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=max(vector_score, 0.0),
                    rank=rank,
                    reason=f"{channel}; vector={vector_score:.3f}; lexical={lexical_score:.3f}",
                )
            )
        return results

    def _connect_and_import(self):
        if not self.endpoint or not self.token:
            raise RuntimeError(
                f"Milvus configuration missing: set {self.config.zilliz_endpoint_env} and {self.config.zilliz_token_env}"
            )
        if not self.is_local_endpoint:
            raise RuntimeError("Online Milvus endpoints are disabled; use the local Docker Milvus service")
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
        except ImportError as exc:
            raise RuntimeError("pymilvus is required for local Milvus integration; install requirements.txt") from exc
        if not self._connected:
            connections.connect(alias=self.config.zilliz_alias, uri=self.endpoint, token=self.token)
            self._connected = True
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType
        self._FieldSchema = FieldSchema
        return utility, Collection

    def _create_collection(self) -> None:
        _, Collection = self._connect_and_import()
        fields = [
            self._FieldSchema(name="id", dtype=self._DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256),
            self._FieldSchema(name="text", dtype=self._DataType.VARCHAR, max_length=8192),
            self._FieldSchema(name="source", dtype=self._DataType.VARCHAR, max_length=1024),
            self._FieldSchema(name="workspace_id", dtype=self._DataType.VARCHAR, max_length=128),
            self._FieldSchema(name="page", dtype=self._DataType.INT64),
            self._FieldSchema(name="block_id", dtype=self._DataType.VARCHAR, max_length=256),
            self._FieldSchema(name="modality", dtype=self._DataType.VARCHAR, max_length=64),
            self._FieldSchema(name="metadata_json", dtype=self._DataType.VARCHAR, max_length=8192),
            self._FieldSchema(name="image_path", dtype=self._DataType.VARCHAR, max_length=1024),
            self._FieldSchema(name=self.vector_field, dtype=self._DataType.FLOAT_VECTOR, dim=self.config.embedding_dimensions),
            self._FieldSchema(name=self.sparse_vector_field, dtype=self._DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = self._CollectionSchema(
            fields=fields,
            description="Education-material OCR multimodal RAG chunks",
            enable_dynamic_field=False,
        )
        Collection(
            name=self.collection_name,
            schema=schema,
            using=self.config.zilliz_alias,
            consistency_level=self.config.zilliz_consistency_level,
        )

    def _ensure_index(self, collection) -> None:
        if not collection.has_index(index_name="vector_index"):
            index_params: Dict[str, Any] = {
                "metric_type": self.config.zilliz_metric_type,
                "index_type": self.config.zilliz_index_type,
                "params": {},
            }
            collection.create_index(field_name=self.vector_field, index_params=index_params, index_name="vector_index")
        field_names = {getattr(field, "name", "") for field in getattr(collection.schema, "fields", [])}
        if self.sparse_vector_field in field_names and not collection.has_index(index_name="sparse_vector_index"):
            collection.create_index(
                field_name=self.sparse_vector_field,
                index_params={"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {}},
                index_name="sparse_vector_index",
            )

    def _delete_existing(self, collection, ids: List[str]) -> None:
        if not ids:
            return
        escaped = [item.replace('"', '\\"') for item in ids]
        expression = "id in [" + ", ".join(f'"{item}"' for item in escaped) + "]"
        collection.delete(expression)

    def _chunks_to_rows(self, chunks: List[DocumentChunk], *, include_workspace_id: bool = False) -> List[Dict[str, Any]]:
        vector_texts = [vector_index_text(chunk) for chunk in chunks]
        batch_method = getattr(self.embedding, "embed_texts", None)
        embed_chunk = getattr(self.embedding, "embed_chunk", None)
        image_indexes = {
            index for index, chunk in enumerate(chunks)
            if callable(embed_chunk) and chunk.image_path and os.path.isfile(chunk.image_path)
        }
        text_indexes = [index for index in range(len(chunks)) if index not in image_indexes]
        text_vectors = batch_method([vector_texts[index] for index in text_indexes]) if callable(batch_method) else [self.embedding.embed_text(vector_texts[index]) for index in text_indexes]
        vectors: List[Optional[List[float]]] = [None] * len(chunks)
        for index, vector in zip(text_indexes, text_vectors):
            vectors[index] = vector
        for index in image_indexes:
            vectors[index] = embed_chunk(chunks[index])
        if any(vector is None for vector in vectors):
            raise RuntimeError("embedding response count does not match knowledge chunk count")
        complete_vectors = [vector or [] for vector in vectors]
        if len(complete_vectors) != len(chunks):
            raise RuntimeError("embedding response count does not match knowledge chunk count")
        return [
            self._chunk_to_row(chunk, vector=vector, include_workspace_id=include_workspace_id)
            for chunk, vector in zip(chunks, complete_vectors)
        ]

    def _chunk_to_row(
        self,
        chunk: DocumentChunk,
        vector: Optional[List[float]] = None,
        *,
        include_workspace_id: bool = False,
    ) -> Dict[str, Any]:
        metadata = dict(chunk.metadata or {})
        metadata.update(
            {
                "parent_id": chunk.parent_id,
                "chunk_level": chunk.chunk_level,
                "confidence": chunk.confidence,
                "needs_review": chunk.needs_review,
            }
        )
        row = {
            "id": chunk.id[:256],
            "text": truncate(chunk.text, 8192),
            "source": truncate(chunk.source, 1024),
            "page": int(chunk.page) if chunk.page is not None else -1,
            "block_id": truncate(chunk.block_id or "", 256),
            "modality": truncate(chunk.modality or "ocr_text", 64),
            "metadata_json": truncate(json.dumps(metadata, ensure_ascii=False), 8192),
            "image_path": truncate(chunk.image_path or "", 1024),
            self.vector_field: vector if vector is not None else self.embedding.embed_text(vector_index_text(chunk)),
            self.sparse_vector_field: self._sparse_vector(
                vector_index_text(chunk)
            ),
        }
        if include_workspace_id:
            row["workspace_id"] = truncate(str(metadata.get("workspace_id") or DEFAULT_WORKSPACE_ID), 128)
        return row

    def _sparse_vector(self, text: str) -> Dict[int, float]:
        method = getattr(self.embedding, "embed_sparse", None)
        values = method(text) if callable(method) else {}
        sparse: Dict[int, float] = {}
        for key, value in values.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                index = int.from_bytes(hashlib.sha256(str(key).encode("utf-8")).digest()[:4], "big") % 2_147_483_647
            sparse[index] = sparse.get(index, 0.0) + float(value)
        # PyMilvus rejects an empty dict for SPARSE_FLOAT_VECTOR fields. Some
        # low-quality or punctuation-only chunks have no lexical terms, so use
        # a zero-weight coordinate to represent an empty sparse vector.
        return sparse or {0: 0.0}


def truncate(value: str, limit: int) -> str:
    text = str(value or "")
    return text[:limit]


def parse_metadata(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def none_if_negative(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return None if number < 0 else number


def empty_to_none(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def terms_for_vector_search(query: str, entities: EntityExtractionResult) -> List[str]:
    terms = split_terms(query)
    for values in entities.entities.values():
        terms.extend(str(value) for value in values)
    return unique_keep_order([term.lower() for term in terms if len(str(term).strip()) >= 2])


def lexical_score_for_chunk(query_terms: List[str], chunk: DocumentChunk) -> float:
    if not query_terms:
        return 0.0
    text = " ".join([chunk.text, chunk.source, json.dumps(chunk.metadata, ensure_ascii=False)]).lower()
    hits = sum(1 for term in query_terms if term and term in text)
    strong_hits = sum(1 for term in query_terms if len(term) >= 3 and term in text)
    return min(1.0, hits / max(len(query_terms), 1) + strong_hits * 0.08)


def unique_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def vector_index_text(chunk: DocumentChunk) -> str:
    """Build the child-only representation used by remote vector indexes."""

    metadata = " ".join(
        str(value) for key, value in (chunk.metadata or {}).items() if key not in {"parent_text", "line_confidences"}
    )
    return " ".join(part for part in (chunk.text, chunk.source, metadata) if part)
