"""BM25 + BGE-M3 dense/sparse retrieval with MMR diversification."""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional

from manufacturing_rag.adapters.embedding import EmbeddingAdapter, HashEmbeddingAdapter, cosine_similarity
from manufacturing_rag.config import DEFAULT_CONFIG, RAGConfig
from manufacturing_rag.query import split_terms, unique_keep_order
from manufacturing_rag.schemas import DEFAULT_WORKSPACE_ID, DocumentChunk, EntityExtractionResult, RetrievalResult


# Query optimization appends these broad guidance words to improve recall.
# They are not evidence anchors: using them for the Milvus preflight can make
# an unrelated active KB look relevant (for example, ``学科文档`` matching a
# math handout when the user asked about scholarships).
GENERIC_ANCHOR_TERMS = {
    "文档",
    "资料",
    "材料",
    "答案",
    "原文",
    "解析",
    "学科",
    "课程",
    "教材",
    "讲义",
    "课件",
    "知识点",
    "概念",
    "主题",
    "内容",
    "信息",
    "什么",
    "哪些",
    "如何",
    "怎么",
    "是否",
    "这个",
    "这份",
}


class HybridRetriever:
    """Three-channel in-memory recall: BM25, BGE-M3 sparse and BGE-M3 dense."""

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        embedding: Optional[EmbeddingAdapter] = None,
        config: RAGConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config
        self.embedding = embedding or HashEmbeddingAdapter(config)
        self.chunks = list(chunks)
        self._search_texts = {chunk.id: searchable_text(chunk) for chunk in self.chunks}
        self._tokens = {chunk.id: split_terms(self._search_texts[chunk.id]) for chunk in self.chunks}
        self._document_frequency = document_frequency(self._tokens.values())
        self._average_length = sum(len(tokens) for tokens in self._tokens.values()) / max(len(self._tokens), 1)
        embed_chunk = getattr(self.embedding, "embed_chunk", None)
        self._chunk_vectors: Dict[str, List[float]] = {
            chunk.id: embed_chunk(chunk) if callable(embed_chunk) and chunk.image_path else self.embedding.embed_text(self._search_texts[chunk.id])
            for chunk in self.chunks
        }
        self._chunk_sparse = {chunk.id: sparse_embedding(self.embedding, self._search_texts[chunk.id]) for chunk in self.chunks}

    def search(self, query: str, entities: EntityExtractionResult, limit: int = 20) -> List[RetrievalResult]:
        if not self.chunks:
            return []
        safe_limit = max(1, int(limit))
        query_terms = split_terms(query)
        for values in entities.entities.values():
            query_terms.extend(str(value) for value in values)
        query_terms = unique_keep_order(str(term).lower() for term in query_terms)
        query_vector = self.embedding.embed_text(query)
        query_sparse = sparse_embedding(self.embedding, query)
        bm25_scores = {
            chunk.id: bm25_score(
                query_terms,
                self._tokens[chunk.id],
                self._document_frequency,
                len(self.chunks),
                self._average_length,
            )
            for chunk in self.chunks
        }
        max_bm25 = max(bm25_scores.values(), default=0.0) or 1.0
        channel_scores: Dict[str, Dict[str, float]] = {
            "bm25": {chunk.id: bm25_scores[chunk.id] / max_bm25 for chunk in self.chunks},
            "sparse": {
                chunk.id: sparse_cosine(query_sparse, self._chunk_sparse[chunk.id])
                for chunk in self.chunks
            },
            "dense": {
                chunk.id: cosine_similarity(query_vector, self._chunk_vectors[chunk.id])
                for chunk in self.chunks
            },
        }
        channel_limit = min(len(self.chunks), max(safe_limit * 4, 20))
        ranked_channels: Dict[str, List[str]] = {}
        for channel, scores in channel_scores.items():
            ranked_channels[channel] = [
                chunk_id
                for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:channel_limit]
                if score > 0
            ]

        rrf_scores: Dict[str, float] = {}
        for channel, chunk_ids in ranked_channels.items():
            route_weight = max(
                0.0,
                {
                    "bm25": self.config.bm25_weight,
                    "sparse": self.config.sparse_weight,
                    "dense": self.config.dense_weight,
                }.get(channel, 1.0),
            )
            for rank, chunk_id in enumerate(chunk_ids, start=1):
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + route_weight / (self.config.rrf_k + rank)

        candidates: List[RetrievalResult] = []
        for chunk in self.chunks:
            text = self._search_texts[chunk.id].lower()
            bm25 = channel_scores["bm25"][chunk.id]
            sparse = channel_scores["sparse"][chunk.id]
            dense = channel_scores["dense"][chunk.id]
            entity_boost = sum(
                0.04
                for values in entities.entities.values()
                if any(str(value).lower() in text for value in values)
            )
            rrf_score = rrf_scores.get(chunk.id, 0.0)
            if rrf_score <= 0 and entity_boost <= 0:
                continue
            score = rrf_score + entity_boost
            if score <= 0:
                continue
            candidates.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    reason=(
                        f"bm25={bm25:.3f}; sparse={sparse:.3f}; dense={dense:.3f}; "
                        f"entity={entity_boost:.3f}; rrf={rrf_score:.5f}"
                    ),
                )
            )
        max_score = max((result.score for result in candidates), default=1.0) or 1.0
        for result in candidates:
            result.score = min(1.0, result.score / max_score)
            result.reason += f"; rrf_normalized={result.score:.3f}"
        candidates.sort(key=lambda result: (-result.score, result.chunk.id))
        diversified = mmr_select(candidates, self._chunk_vectors, limit=safe_limit, lambda_mult=self.config.mmr_lambda)
        for rank, result in enumerate(diversified, start=1):
            result.rank = rank
        return diversified

    def search_many(
        self,
        queries: Iterable[str],
        entities: EntityExtractionResult,
        limit: int = 20,
    ) -> List[RetrievalResult]:
        """Recall each query variant and fuse the routes by rank using RRF."""

        query_list = unique_keep_order(query for query in queries)
        if not query_list:
            return []
        if len(query_list) == 1:
            return self.search(query_list[0], entities, limit=limit)
        route_results = [self.search(query, entities, limit=limit) for query in query_list]
        return rrf_fuse_results(
            route_results,
            limit=limit,
            rrf_k=self.config.rrf_k,
            route_names=[f"query_{index}" for index in range(1, len(route_results) + 1)],
        )


class InMemoryMultimodalRetriever(HybridRetriever):
    """Backward-compatible name for the full local hybrid retriever."""


class MilvusHybridRetriever:
    """Fuse Milvus dense recall with local BM25/sparse recall before CrossEncoder."""

    def __init__(
        self,
        remote: object,
        chunks: Iterable[DocumentChunk],
        embedding: EmbeddingAdapter,
        config: RAGConfig,
        cache_callback: Optional[Callable[[List[DocumentChunk]], None]] = None,
        workspace_id: Optional[str] = None,
        workspace_ids: Optional[Iterable[str]] = None,
        strict_workspace_metadata: bool = False,
    ) -> None:
        self.remote = remote
        self.cache_callback = cache_callback
        active_chunks = list(chunks)
        # The local KB is the active source of truth for this process.  A
        # Keep the IDs and local searchable text so current-process uploads
        # can use a fast isolated path. When the local snapshot is incomplete
        # after a restart, the remote collection remains a valid source of
        # uploaded documents and is consulted as a fallback.
        self.active_chunk_ids = frozenset(chunk.id for chunk in active_chunks)
        self.active_search_texts = {
            chunk.id: searchable_text(chunk).lower()
            for chunk in active_chunks
        }
        self.local = HybridRetriever(active_chunks, embedding=embedding, config=config)
        self.config = config
        resolved_workspace_ids = set(workspace_ids or ())
        if workspace_id is not None:
            resolved_workspace_ids.add(workspace_id)
        self.workspace_ids = frozenset(resolved_workspace_ids)
        self.strict_workspace_metadata = strict_workspace_metadata

    def search(self, query: str, entities: EntityExtractionResult, limit: int = 20) -> List[RetrievalResult]:
        local_results = self.local.search(query, entities, limit=max(limit, self.config.candidate_k))
        local_has_anchor = bool(self.active_chunk_ids) and has_active_lexical_anchor(
            query, entities, self.active_search_texts.values()
        )
        # A direct local hit is already source-grounded. Keep the fast path
        # local and reserve remote recall for harder or ambiguous cases.
        if local_results and any(
            has_active_lexical_anchor(query, entities, [searchable_text(result.chunk).lower()])
            for result in local_results
        ):
            return local_results[:limit]
        try:
            if self.strict_workspace_metadata:
                remote_results = self.remote.search(
                    query,
                    entities,
                    limit=max(limit, self.config.candidate_k),
                    workspace_ids=self.workspace_ids,
                )
            else:
                remote_results = self.remote.search(query, entities, limit=max(limit, self.config.candidate_k))
        except Exception:
            if local_results:
                return local_results[:limit]
            raise
        if self.workspace_ids:
            remote_results = [
                result
                for result in remote_results
                if (
                    (not self.strict_workspace_metadata or result.chunk.metadata.get("workspace_id"))
                    and str(result.chunk.metadata.get("workspace_id") or DEFAULT_WORKSPACE_ID) in self.workspace_ids
                )
            ]
        # When local state was rebuilt after a restart, the persistent vector
        # collection can contain valid uploaded documents that are not yet in
        # the in-memory snapshot. Allow those rows to reach the pipeline; its
        # lexical-anchor and source-grounding filters still reject unrelated
        # near-misses. If local state has an anchor, keep the old isolation
        # rule and only fuse rows belonging to the active snapshot.
        if local_has_anchor:
            remote_results = [
                result for result in remote_results
                if result.chunk.id in self.active_chunk_ids
            ]
        return rrf_fuse_results(
            [local_results, remote_results],
            limit=limit,
            rrf_k=self.config.rrf_k,
            route_names=["local_hybrid", "milvus_dense"],
        )

    def cache_chunks(self, chunks: Iterable[DocumentChunk]) -> List[DocumentChunk]:
        """Add trusted remote contexts to the local working index."""

        current_by_id = {chunk.id: chunk for chunk in self.local.chunks}
        changed: List[DocumentChunk] = []
        for chunk in chunks:
            if not chunk.id:
                continue
            if current_by_id.get(chunk.id) == chunk:
                continue
            current_by_id[chunk.id] = chunk
            changed.append(chunk)
        if not changed:
            return []
        cached_chunks = list(current_by_id.values())
        self.local = HybridRetriever(cached_chunks, embedding=self.local.embedding, config=self.config)
        self.active_chunk_ids = frozenset(chunk.id for chunk in cached_chunks)
        self.active_search_texts = {
            chunk.id: searchable_text(chunk).lower()
            for chunk in cached_chunks
        }
        if self.cache_callback:
            self.cache_callback(changed)
        return changed

    def search_many(
        self,
        queries: Iterable[str],
        entities: EntityExtractionResult,
        limit: int = 20,
    ) -> List[RetrievalResult]:
        query_list = unique_keep_order(query for query in queries)
        if not query_list:
            return []
        route_results = [self.search(query, entities, limit=limit) for query in query_list]
        return rrf_fuse_results(
            route_results,
            limit=limit,
            rrf_k=self.config.rrf_k,
            route_names=[f"query_{index}" for index in range(1, len(route_results) + 1)],
        )


def searchable_text(chunk: DocumentChunk) -> str:
    metadata_text = " ".join(
        str(value) for key, value in chunk.metadata.items() if key not in {"parent_text", "line_confidences"}
    )
    # Parent text is deliberately excluded from the index representation.  It
    # is restored only after child recall and reranking (Small-to-Big).
    return " ".join(part for part in (chunk.text, chunk.source, metadata_text) if part)


def has_active_lexical_anchor(
    query: str,
    entities: EntityExtractionResult,
    searchable_texts: Iterable[str],
) -> bool:
    """Return whether the active KB contains a term-level support anchor."""

    query_terms = split_terms(query)
    for values in entities.entities.values():
        query_terms.extend(str(value) for value in values)
    terms = unique_keep_order(
        str(term).strip().lower()
        for term in query_terms
        if len(str(term).strip()) >= 2
        and str(term).strip().lower() not in GENERIC_ANCHOR_TERMS
    )
    if not terms:
        return False
    # Prefer a complete 3+ character phrase when one exists. Matching a
    # two-character substring such as ``事务`` would falsely mark ``事务锁``
    # as present in an unrelated handbook page.
    long_terms = [term for term in terms if len(term) >= 3]
    question_shape = any(marker in query for marker in ("是什么", "什么是", "定义", "有哪些", "包括", "如何", "怎么", "是否"))
    anchor_terms = terms if not long_terms or question_shape else long_terms
    return any(term in text for text in searchable_texts for term in anchor_terms)


def document_frequency(token_lists: Iterable[List[str]]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for tokens in token_lists:
        frequency.update(set(tokens))
    return frequency


def bm25_score(
    query_terms: List[str],
    document_terms: List[str],
    frequencies: Counter[str],
    document_count: int,
    average_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_terms or not document_terms:
        return 0.0
    term_counts = Counter(document_terms)
    score = 0.0
    length_ratio = len(document_terms) / max(average_length, 1.0)
    for term in query_terms:
        frequency = term_counts.get(term, 0)
        if not frequency:
            continue
        doc_frequency = frequencies.get(term, 0)
        inverse_document_frequency = math.log(1.0 + (document_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
        score += inverse_document_frequency * frequency * (k1 + 1.0) / (
            frequency + k1 * (1.0 - b + b * length_ratio)
        )
    return score


def sparse_embedding(adapter: EmbeddingAdapter, text: str) -> Dict[str, float]:
    method = getattr(adapter, "embed_sparse", None)
    if callable(method):
        return {str(key): float(value) for key, value in method(text).items()}
    return HashEmbeddingAdapter().embed_sparse(text)


def sparse_cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values())) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right.values())) or 1.0
    return max(0.0, numerator / (left_norm * right_norm))


def mmr_select(
    candidates: List[RetrievalResult],
    vectors: Dict[str, List[float]],
    limit: int,
    lambda_mult: float,
) -> List[RetrievalResult]:
    if not candidates or limit <= 0:
        return []
    remaining = list(candidates)
    selected: List[RetrievalResult] = []
    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda item: lambda_mult * item.score
            - (1.0 - lambda_mult)
            * max(
                (cosine_similarity(vectors.get(item.chunk.id, []), vectors.get(chosen.chunk.id, [])) for chosen in selected),
                default=0.0,
            ),
        )
        best.reason += "; mmr_selected"
        selected.append(best)
        remaining.remove(best)
    return selected


def rrf_fuse_results(
    result_lists: Iterable[List[RetrievalResult]],
    limit: int,
    rrf_k: int = 60,
    route_names: Optional[List[str]] = None,
    route_weights: Optional[List[float]] = None,
) -> List[RetrievalResult]:
    """Fuse ranked result lists without comparing incompatible raw scores."""

    safe_limit = max(1, int(limit))
    safe_rrf_k = max(1, int(rrf_k))
    lists = list(result_lists)
    if not any(lists):
        return []
    names = route_names or [f"route_{index}" for index in range(1, len(lists) + 1)]
    fused: Dict[str, RetrievalResult] = {}
    raw_scores: Dict[str, float] = {}
    route_details: Dict[str, List[str]] = {}
    for route_index, results in enumerate(lists):
        if not results:
            continue
        route_name = names[route_index] if route_index < len(names) else f"route_{route_index + 1}"
        route_weight = max(0.0, float(route_weights[route_index])) if route_weights and route_index < len(route_weights) else 1.0
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id
            raw_scores[chunk_id] = raw_scores.get(chunk_id, 0.0) + route_weight / (safe_rrf_k + rank)
            route_details.setdefault(chunk_id, []).append(f"{route_name}@{rank}")
            if chunk_id not in fused:
                fused[chunk_id] = RetrievalResult(
                    chunk=result.chunk,
                    score=0.0,
                    reason=result.reason,
                )
    ranked_ids = sorted(raw_scores, key=lambda chunk_id: (-raw_scores[chunk_id], chunk_id))[:safe_limit]
    normalizer = max((raw_scores[chunk_id] for chunk_id in ranked_ids), default=1.0) or 1.0
    ranked: List[RetrievalResult] = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        result = fused[chunk_id]
        result.score = min(1.0, raw_scores[chunk_id] / normalizer)
        result.rank = rank
        result.reason = (
            f"{result.reason}; rrf={raw_scores[chunk_id]:.5f}; "
            f"rrf_routes={','.join(route_details[chunk_id])}"
        )
        ranked.append(result)
    return ranked
