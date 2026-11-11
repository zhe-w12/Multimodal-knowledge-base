"""End-to-end four-model education RAG orchestration."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from manufacturing_rag.adapters.llm import RuleBasedLLMRouter, build_llm_router
from manufacturing_rag.adapters.embedding import HashEmbeddingAdapter, build_embedding_adapter
from manufacturing_rag.adapters.ocr import OCRStructuredLoader
from manufacturing_rag.adapters.query_optimizer import AdaptiveQueryOptimizer
from manufacturing_rag.adapters.rerank import RuleBasedReranker, build_reranker
from manufacturing_rag.adapters.retriever import (
    HybridRetriever,
    MilvusHybridRetriever,
    has_active_lexical_anchor,
    rrf_fuse_results,
    searchable_text,
)
from manufacturing_rag.adapters.vector_store import MilvusVectorStore
from manufacturing_rag.config import DEFAULT_CONFIG, RAGConfig
from manufacturing_rag.debug_log import retrieval_results_for_debug, write_debug_log
from manufacturing_rag.query import split_terms
from manufacturing_rag.schemas import (
    DocumentChunk,
    EntityExtractionResult,
    HallucinationCheck,
    IntentResult,
    RAGResponse,
    SafetyResult,
)
from manufacturing_rag.security import SafetyGuard


class ManufacturingRAGPipeline:
    """Strict, source-grounded RAG pipeline for education materials."""

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        config: RAGConfig = DEFAULT_CONFIG,
        retriever: Optional[object] = None,
        reranker: Optional[RuleBasedReranker] = None,
        llm_router: Optional[RuleBasedLLMRouter] = None,
        safety_guard: Optional[SafetyGuard] = None,
        remote_cache_callback: Optional[Callable[[List[DocumentChunk]], None]] = None,
        workspace_id: Optional[str] = None,
        workspace_ids: Optional[Iterable[str]] = None,
        strict_workspace_metadata: bool = False,
    ) -> None:
        self.config = config
        self.chunks = list(chunks)
        embedding = build_embedding_adapter(config)
        if retriever is not None:
            self.retriever = retriever
        elif config.vector_store_provider in {"zilliz", "milvus"}:
            checked_embedding = build_embedding_adapter(config, check_dimensions=True)
            remote = MilvusVectorStore(config=config, embedding=checked_embedding)
            # The remote vector store owns semantic retrieval in this mode.
            # Keep the local BM25/sparse fallback deterministic; otherwise a
            # PDF upload rebuilds the local index with one cloud embedding
            # request per chunk and the browser eventually reports
            # ``Failed to fetch`` while the request is still running.
            local_embedding = (
                HashEmbeddingAdapter(config)
                if config.embedding_provider in {"bailian", "dashscope", "aliyun"}
                else embedding
            )
            self.retriever = MilvusHybridRetriever(
                remote,
                self.chunks,
                embedding=local_embedding,
                config=config,
                cache_callback=remote_cache_callback,
                workspace_id=workspace_id,
                workspace_ids=workspace_ids,
                strict_workspace_metadata=strict_workspace_metadata,
            )
        else:
            self.retriever = HybridRetriever(self.chunks, embedding=embedding, config=config)
        self.reranker = reranker or build_reranker(config)
        self.llm_router = llm_router or build_llm_router(config)
        self.query_optimizer = AdaptiveQueryOptimizer(self.llm_router, config)
        self.safety_guard = safety_guard or SafetyGuard(config)

    def cache_retrieved_contexts(self, contexts: Iterable[object]) -> None:
        """Promote validated remote contexts into the local working index."""

        if not isinstance(self.retriever, MilvusHybridRetriever):
            return
        chunks = [context.chunk for context in contexts if getattr(context, "chunk", None)]
        changed = self.retriever.cache_chunks(chunks)
        if not changed:
            return
        chunks_by_id = {chunk.id: chunk for chunk in self.chunks}
        chunks_by_id.update({chunk.id: chunk for chunk in changed})
        self.chunks = list(chunks_by_id.values())

    @classmethod
    def from_jsonl(cls, paths: Iterable[str], config: RAGConfig = DEFAULT_CONFIG) -> "ManufacturingRAGPipeline":
        loader = OCRStructuredLoader()
        return cls(loader.load_many(paths), config=config)

    def answer(
        self,
        question: str,
        conversation_context: str = "",
        rerank_rules: Optional[dict] = None,
        on_token: Optional[Callable[[str], None]] = None,
        hallucination_threshold: float = 0.75,
        on_status: Optional[Callable[[str], None]] = None,
        classified_intent: Optional[IntentResult] = None,
    ) -> RAGResponse:
        if on_status and classified_intent is None:
            on_status("正在识别问题类型…")
        intent = classified_intent or self.classify_intent(question, conversation_context)
        safety = self.safety_guard.assess_question(question)
        empty_entities = EntityExtractionResult(raw_query=question, entities={})

        if not intent.allowed or not safety.allowed:
            response_safety = safety if not safety.allowed else SafetyResult(False, "high", [intent.category], intent.reason)
            return RAGResponse(
                question=question,
                answer="该请求未通过学科文档问答安全审核，无法进入检索或生成链路。",
                intent=intent,
                safety=response_safety,
                entities=empty_entities,
                hallucination_check=HallucinationCheck(True, [], 1.0),
                notices=[intent.reason, response_safety.reason],
            )

        if intent.category == "chit_chat":
            if on_status:
                on_status("这是闲聊问题，正在由对话模型回复…")
            chat_stream = getattr(self.llm_router, "generate_chat_response_stream", None)
            if on_token and callable(chat_stream):
                chat_answer = chat_stream(question, conversation_context, on_token)
            else:
                chat_generator = getattr(self.llm_router, "generate_chat_response", None)
                if callable(chat_generator):
                    chat_answer = chat_generator(question, conversation_context)
                else:
                    chat_answer = "当前对话模型不可用，请稍后重试。"
                    if on_token:
                        on_token(chat_answer)
            chat_answer = self.safety_guard.redact(chat_answer)
            answer_source = str(getattr(self.llm_router, "last_answer_source", "remote_llm"))
            model_notice = (
                f"闲聊回答由 {self.config.llm_provider} / {self.config.llm_model} 生成"
                if answer_source == "remote_llm"
                else "闲聊回答由本地兜底生成"
            )
            return RAGResponse(
                question=question,
                answer=chat_answer,
                intent=intent,
                safety=safety,
                entities=empty_entities,
                hallucination_check=HallucinationCheck(True, [], 1.0),
                notices=["闲聊路线：本次没有检索知识库资料", model_notice],
            )

        retrieval_question = contextualize_question(question, conversation_context)
        entities = self.query_optimizer.extract_entities(retrieval_question)
        query_variants = self.query_optimizer.expand_queries(retrieval_question, intent, entities)
        if not query_variants:
            query_variants = [retrieval_question]
        # The first variant is always the user's wording; expose the first
        # rewritten form as the primary query for diagnostics and compatibility.
        optimized_query = query_variants[1] if len(query_variants) > 1 else query_variants[0]
        candidates = retrieve_with_query_variants(
            self.retriever,
            query_variants,
            entities,
            limit=self.config.candidate_k,
            rrf_k=self.config.rrf_k,
        )
        if isinstance(self.retriever, MilvusHybridRetriever):
            candidates = filter_unanchored_candidates(
                retrieval_question,
                entities,
                candidates,
                active_chunks=self.chunks,
            )
        if on_status:
            on_status("正在筛选与问题最相关的资料…")
        write_debug_log(
            self.config,
            "retrieval_candidates",
            question=question,
            optimized_query=optimized_query,
            query_variants=query_variants,
            candidate_count=len(candidates),
            results=retrieval_results_for_debug(candidates, text_chars=self.config.debug_log_text_chars),
        )
        reranked_candidates = self.reranker.rerank(
            optimized_query,
            candidates,
            top_k=max(self.config.top_k, len(candidates)),
            threshold=0.0,
        )
        contexts = apply_configurable_rerank_rules(reranked_candidates, rerank_rules)
        contexts = [result for result in contexts if result.score >= self.config.rerank_threshold]
        contexts = contexts[: self.config.top_k]
        for rank, result in enumerate(contexts, start=1):
            result.rank = rank
        contexts = filter_low_quality_contexts(retrieval_question, contexts)
        self.cache_retrieved_contexts(contexts)
        contexts = expand_parent_contexts(contexts)
        write_debug_log(
            self.config,
            "retrieval_contexts_after_rerank",
            question=question,
            optimized_query=optimized_query,
            query_variants=query_variants,
            context_count=len(contexts),
            threshold=self.config.rerank_threshold,
            results=retrieval_results_for_debug(contexts, text_chars=self.config.debug_log_text_chars),
        )
        sufficient, sufficiency_reason = self.llm_router.assess_context_sufficiency(retrieval_question, contexts)
        write_debug_log(
            self.config,
            "context_sufficiency",
            question=question,
            optimized_query=optimized_query,
            sufficient=sufficient,
            reason=sufficiency_reason,
            context_count=len(contexts),
        )
        if not sufficient:
            if on_status:
                on_status("正在请 AI 确认缺少的信息…")
            clarification_generator = getattr(self.llm_router, "generate_clarifying_question", None)
            if callable(clarification_generator):
                clarification = clarification_generator(question, contexts, sufficiency_reason)
            else:
                clarification = build_insufficient_follow_up(question, contexts, sufficiency_reason)
            if on_token:
                for start in range(0, len(clarification), 24):
                    on_token(clarification[start : start + 24])
            return RAGResponse(
                question=question,
                answer=clarification,
                intent=intent,
                safety=safety,
                optimized_query=optimized_query,
                query_variants=query_variants,
                entities=entities,
                sources=[result.to_source_ref() for result in contexts],
                retrieval_confidence=max((result.score for result in contexts), default=0.0),
                hallucination_check=HallucinationCheck(True, [], 1.0),
                notices=[sufficiency_reason],
            )

        if on_status:
            on_status("DeepSeek 正在根据资料组织答案…")
        stream_generator = getattr(self.llm_router, "generate_answer_stream", None)
        if on_token and callable(stream_generator):
            draft = stream_generator(retrieval_question, contexts, on_token)
        else:
            draft = self.llm_router.generate_answer(retrieval_question, contexts)
            if on_token:
                on_token(draft)
        answer_source = str(getattr(self.llm_router, "last_answer_source", "local_rule"))
        redacted_answer = self.safety_guard.redact(draft)
        redacted = redacted_answer != draft
        post_safety = self.safety_guard.post_audit(redacted_answer)
        source_texts = source_texts_for_validation(contexts)
        if on_status:
            on_status("正在核对答案与资料的一致性…")
        hallucination = apply_hallucination_threshold(
            self.llm_router.hallucination_check(redacted_answer, source_texts),
            hallucination_threshold,
            source_texts,
        )
        answer_repaired = False
        repair_generator = getattr(self.llm_router, "repair_answer", None)
        if not hallucination.passed and answer_source == "remote_llm" and callable(repair_generator):
            try:
                if on_status:
                    on_status("正在根据核验结果完善答案…")
                repaired_draft = repair_generator(
                    retrieval_question,
                    contexts,
                    redacted_answer,
                    hallucination.issues,
                )
                repaired_answer = self.safety_guard.redact(repaired_draft)
                repaired_safety = self.safety_guard.post_audit(repaired_answer)
                repaired_check = apply_hallucination_threshold(
                    self.llm_router.hallucination_check(repaired_answer, source_texts),
                    hallucination_threshold,
                    source_texts,
                )
                if repaired_check.passed and repaired_safety.allowed:
                    redacted_answer = repaired_answer
                    redacted = repaired_answer != repaired_draft
                    post_safety = repaired_safety
                    hallucination = repaired_check
                    answer_repaired = True
                else:
                    if not repaired_safety.allowed:
                        post_safety = repaired_safety
                    if not repaired_check.passed:
                        hallucination = repaired_check
            except Exception:  # noqa: BLE001 - keep the safe refusal if correction cannot be validated.
                pass
        notices: List[str] = [sufficiency_reason]
        if answer_source == "remote_llm":
            notices.append(f"答案已由 {self.config.llm_provider} / {self.config.llm_model} 大模型生成")
        elif answer_source == "rule_fallback":
            notices.append(f"{self.config.llm_provider} / {self.config.llm_model} 调用未成功，当前答案由本地规则兜底生成")
        else:
            notices.append("答案由本地规则生成")
        write_debug_log(
            self.config,
            "answer_generation",
            question=question,
            answer_source=answer_source,
            remote_error_present=bool(getattr(self.llm_router, "last_remote_error", "")),
        )
        if redacted:
            notices.append("输出已执行涉密/凭据类内容脱敏")
        if not post_safety.allowed:
            notices.append(post_safety.reason)
        if answer_repaired:
            notices.append("答案已根据资料校验结果重新整理")
        if not hallucination.passed:
            redacted_answer = "幻觉自检未通过：生成内容包含资料未支撑的信息，已停止输出结论。"
            notices.extend(hallucination.issues)

        return RAGResponse(
            question=question,
            answer=redacted_answer,
            intent=intent,
            safety=post_safety if not post_safety.allowed else safety,
            optimized_query=optimized_query,
            query_variants=query_variants,
            entities=entities,
            sources=[result.to_source_ref() for result in contexts],
            retrieval_confidence=max((result.score for result in contexts), default=0.0),
            hallucination_check=hallucination,
            redacted=redacted,
            notices=notices,
        )

    def classify_intent(self, question: str, conversation_context: str = "") -> IntentResult:
        """Classify the user's message before choosing chat or RAG processing."""

        return self.query_optimizer.classify_intent(question, conversation_context)


def filter_low_quality_contexts(question: str, contexts):
    """Keep answer evidence focused when a clean topic match exists.

    PDF tables of contents often contain the exact title but no usable body
    text.  If a reviewed/usable chunk contains the longest topic phrase, the
    TOC hit and distant one-word mentions are not valid answer evidence. Keep
    adjacent pages and the same parent so a section can still span chunks.
    """

    if len(contexts) < 2:
        return contexts
    generic_terms = {
        "文档", "资料", "材料", "答案", "原文", "解析", "学科", "课程", "教材",
        "讲义", "课件", "知识点", "概念", "主题", "内容", "信息", "什么", "哪些",
        "如何", "怎么", "是否", "这个", "这份",
    }
    topic_terms = [
        term.lower()
        for term in split_terms(question)
        if len(term.strip()) >= 3 and term.lower() not in generic_terms
    ]
    if not topic_terms:
        return contexts
    topic_phrase = max(topic_terms, key=len)
    usable = [result for result in contexts if not result.chunk.needs_review]
    primary = [
        result for result in usable
        if topic_phrase in compact_text(result.chunk.text)
    ]
    if not primary:
        return contexts
    primary_parent_ids = {
        result.chunk.parent_id or str(result.chunk.metadata.get("parent_id") or "")
        for result in primary
    }
    primary_pages = {
        int(result.chunk.page)
        for result in primary
        if result.chunk.page is not None and str(result.chunk.page).isdigit()
    }
    filtered = []
    for result in contexts:
        chunk = result.chunk
        if chunk.needs_review:
            continue
        parent_id = chunk.parent_id or str(chunk.metadata.get("parent_id") or "")
        page_is_adjacent = any(
            chunk.page is not None and str(chunk.page).isdigit() and abs(int(chunk.page) - page) <= 1
            for page in primary_pages
        )
        same_parent = bool(parent_id and parent_id in primary_parent_ids)
        if topic_phrase in compact_text(chunk.text) or same_parent or page_is_adjacent:
            filtered.append(result)
    return filtered or contexts


def source_texts_for_validation(contexts: Iterable[object]) -> List[str]:
    """Include trace metadata when validating generated numeric citations."""

    source_texts = []
    for result in contexts:
        chunk = result.chunk
        labels = [f"来源：{chunk.source or chunk.id}"]
        if chunk.page is not None:
            labels.append(f"第{chunk.page}页")
        if chunk.block_id:
            labels.append(f"块：{chunk.block_id}")
        source_texts.append(" | ".join(labels) + "\n" + chunk.text)
    return source_texts


def apply_hallucination_threshold(
    check: HallucinationCheck,
    threshold: float,
    source_texts: Iterable[str],
) -> HallucinationCheck:
    """Apply the user's confidence threshold to both positive and negative checks."""

    safe_threshold = min(0.95, max(0.0, float(threshold)))
    confidence = min(1.0, max(0.0, float(check.confidence)))
    issues = list(check.issues)
    if not any(str(text).strip() for text in source_texts):
        return HallucinationCheck(False, issues or ["缺少可核验来源文本"], confidence)
    if confidence < safe_threshold:
        reason = f"来源核验置信度为 {confidence:.0%}，低于设置阈值 {safe_threshold:.0%}"
        return HallucinationCheck(False, [reason, *issues], confidence)
    if not check.passed:
        warning = "当前阈值已放宽来源校验；请核对参考资料后使用此回答。"
        return HallucinationCheck(True, [warning, *issues], confidence)
    return HallucinationCheck(True, issues, confidence)


def filter_unanchored_candidates(
    question: str,
    entities: EntityExtractionResult,
    candidates,
    active_chunks: Iterable[DocumentChunk] = (),
):
    """Drop remote near-misses when at least one exact topic anchor exists.

    Dense retrieval can return semantically adjacent pages for an unsupported
    short term.  Once a complete query phrase is present in any candidate,
    candidates without that phrase are not safe evidence for generation.
    """

    if not candidates:
        return candidates
    anchored = [
        result
        for result in candidates
        if has_active_lexical_anchor(
            question,
            entities,
            [searchable_text(result.chunk).lower()],
        )
    ]
    if anchored:
        return anchored
    active_texts = [searchable_text(chunk).lower() for chunk in active_chunks]
    if active_texts and not has_active_lexical_anchor(question, entities, active_texts):
        return []
    return candidates


def compact_text(value: str) -> str:
    return "".join(str(value or "").lower().split())


def apply_configurable_rerank_rules(candidates, rules: Optional[dict] = None):
    """Apply user-configured keyword boosts and exclusions after model reranking."""

    rule_values = rules if isinstance(rules, dict) else {}

    def normalized_terms(key: str) -> List[str]:
        raw_terms = rule_values.get(key, [])
        if not isinstance(raw_terms, (list, tuple)):
            return []
        terms = []
        for value in raw_terms[:20]:
            term = str(value or "").strip().lower()[:64]
            if term and term not in terms:
                terms.append(term)
        return terms

    boost_terms = normalized_terms("boost_terms")
    exclude_terms = normalized_terms("exclude_terms")
    reranked = []
    for result in candidates:
        searchable = compact_text(f"{result.chunk.text} {result.chunk.source}")
        excluded = [term for term in exclude_terms if compact_text(term) in searchable]
        if excluded:
            result.reason += f"; excluded_by_user_rule={','.join(excluded)}"
            continue
        boosted = [term for term in boost_terms if compact_text(term) in searchable]
        if boosted:
            result.score = min(1.0, result.score + min(0.18, 0.06 * len(boosted)))
            result.reason += f"; user_rule_boost={','.join(boosted)}"
        reranked.append(result)

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked


def expand_parent_contexts(contexts):
    """Restore the larger parent text only after child-level retrieval and reranking."""

    from manufacturing_rag.schemas import RetrievalResult

    expanded = []
    seen_parents = set()
    for result in contexts:
        parent_id = result.chunk.parent_id or str(result.chunk.metadata.get("parent_id") or "")
        parent_text = str(result.chunk.metadata.get("parent_text") or "").strip()
        if parent_id and parent_id in seen_parents:
            continue
        if parent_id:
            seen_parents.add(parent_id)
        # Low-confidence PDF text is often a table of contents or layout
        # noise. Expanding it to a full parent can drown out the actual
        # evidence, so keep the reviewed child text as-is.
        if result.chunk.needs_review:
            expanded.append(result)
            continue
        if not parent_text or parent_text == result.chunk.text:
            expanded.append(result)
            continue
        chunk = DocumentChunk(
            id=result.chunk.id,
            text=parent_text,
            source=result.chunk.source,
            page=result.chunk.page,
            block_id=result.chunk.block_id,
            modality=result.chunk.modality,
            coordinates=result.chunk.coordinates,
            metadata=dict(result.chunk.metadata),
            image_path=result.chunk.image_path,
            parent_id=result.chunk.parent_id,
            chunk_level="parent_context",
            confidence=result.chunk.confidence,
            needs_review=result.chunk.needs_review,
        )
        expanded.append(RetrievalResult(chunk=chunk, score=result.score, rank=result.rank, reason=result.reason + "; parent_expanded"))
    return expanded


def retrieve_with_query_variants(
    retriever: object,
    query_variants: Iterable[str],
    entities: EntityExtractionResult,
    limit: int,
    rrf_k: int = 60,
):
    """Run Multi-Query recall and fuse each route by rank.

    The helper also supports lightweight custom retrievers used by tests and
    integrations that only implement ``search``.
    """

    variants = list(dict.fromkeys(str(query).strip() for query in query_variants if str(query).strip()))
    if not variants:
        return []
    search_many = getattr(retriever, "search_many", None)
    if callable(search_many):
        return search_many(variants, entities, limit=limit)
    search = getattr(retriever, "search")
    if len(variants) == 1:
        return search(variants[0], entities, limit=limit)
    route_results = [search(query, entities, limit=limit) for query in variants]
    return rrf_fuse_results(
        route_results,
        limit=limit,
        rrf_k=rrf_k,
        route_names=[f"query_{index}" for index in range(1, len(route_results) + 1)],
    )


EducationRAGPipeline = ManufacturingRAGPipeline


def build_insufficient_follow_up(question: str, contexts, reason: str) -> str:
    """Turn an insufficient-evidence result into one useful follow-up question."""

    normalized = " ".join(str(question or "").split()).strip().rstrip("？?。.!！")
    focus = normalized if len(normalized) <= 40 else normalized[:39].rstrip() + "…"
    reason_text = str(reason or "")

    if "关键对象" in reason_text:
        subject = reason_text.split(":", 1)[-1].split("：", 1)[-1].strip().rstrip("。")[:24]
        if subject:
            return f"你提到的“{subject}”，具体指哪个课程、章节或题目？"
    if "具体数据" in reason_text:
        return "你要核实的是哪项数据？能补充一下年份或适用范围吗？"
    if any(term in normalized for term in ("学校", "校内", "本校")) and any(term in normalized for term in ("名单", "哪些", "谁")):
        return "你想查的是哪所学校、哪一年度或哪个项目的名单？"
    if any(term in normalized for term in ("学校", "校内", "本校", "奖学金", "助学金", "政策")):
        return "你指的是哪所学校、哪一年度或哪一项政策？"
    if any(term in normalized for term in ("题", "例题", "作业", "练习")):
        return f"关于“{focus}”，你能补充课程、章节或具体题目吗？"
    if contexts:
        return f"你最想确认“{focus}”中的哪项条件或适用范围？"
    if focus:
        return f"关于“{focus}”，你能补充相关课程、章节或具体范围吗？"
    return "你想查询哪门课程、哪个章节或哪份资料？"


def contextualize_question(question: str, conversation_context: str) -> str:
    """Resolve short or referential follow-ups with recent session context."""

    if not conversation_context:
        return question
    compact = question.strip()
    follow_up_markers = (
        "这个", "这项", "这条", "它", "上述", "前面", "刚才", "那么", "那", "该项", "其中",
        "还有", "另外", "继续", "展开", "具体说说", "为什么", "怎么改", "如何呢", "第二个", "第一项",
    )
    needs_context = len(compact) <= 14 or any(term in compact for term in follow_up_markers)
    if not needs_context:
        return question
    return f"会话上下文：{conversation_context}\n当前问题：{question}"
