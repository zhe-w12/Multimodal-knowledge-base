import base64

from fastapi.testclient import TestClient

from manufacturing_rag.adapters.embedding import HashEmbeddingAdapter
from manufacturing_rag.adapters.llm import RuleBasedLLMRouter
from manufacturing_rag.adapters.query_optimizer import assess_query_difficulty
from manufacturing_rag.adapters.retriever import HybridRetriever, MilvusHybridRetriever
from manufacturing_rag.api import create_app
from manufacturing_rag.chunking import prepare_chunks
from manufacturing_rag.config import RAGConfig
from manufacturing_rag.pipeline import filter_low_quality_contexts, filter_unanchored_candidates, source_texts_for_validation
from manufacturing_rag.query import RuleBasedQueryProcessor
from manufacturing_rag.schemas import DocumentChunk, EntityExtractionResult, RetrievalResult


def offline_config(**overrides):
    values = {
        "llm_provider": "rule",
        "embedding_provider": "hash",
        "rerank_provider": "rule",
        "vector_store_provider": "memory",
        "query_classifier_provider": "rule",
        "query_agent_provider": "rule",
        "redis_enabled": False,
        "mysql_enabled": False,
        "embedding_dimensions": 64,
    }
    values.update(overrides)
    return RAGConfig(**values)


def test_parent_child_chunking_retains_context_and_marks_low_confidence():
    config = offline_config(parent_chunk_chars=120, child_chunk_chars=48, child_chunk_overlap=8, min_etl_confidence=0.8)
    raw = DocumentChunk(
        id="lesson",
        text="第一节介绍函数极限的定义。" * 12,
        source="lesson.pdf",
        confidence=0.6,
    )

    chunks, report = prepare_chunks([raw], config)

    assert len(chunks) > 1
    assert all(chunk.parent_id and chunk.metadata["parent_text"] for chunk in chunks)
    assert all(chunk.needs_review for chunk in chunks)
    assert report.review_chunks == 1


def test_hybrid_retriever_exposes_all_three_recall_scores():
    config = offline_config()
    chunks = [
        DocumentChunk(id="matrix", text="矩阵可逆的充要条件是行列式不等于零。", source="linear-algebra.pdf"),
        DocumentChunk(id="limit", text="函数极限描述函数值的变化趋势。", source="calculus.pdf"),
    ]
    retriever = HybridRetriever(chunks, embedding=HashEmbeddingAdapter(config), config=config)

    results = retriever.search("矩阵可逆条件", EntityExtractionResult(raw_query="矩阵可逆条件"), limit=2)

    assert results[0].chunk.id == "matrix"
    assert all(name in results[0].reason for name in ("bm25=", "sparse=", "dense=", "mmr_selected"))


def test_milvus_retriever_uses_remote_fallback_but_pipeline_filters_unanchored_rows():
    config = offline_config()
    active = DocumentChunk(id="active", text="函数极限描述函数值的变化趋势。", source="current.pdf")
    stale = DocumentChunk(id="stale", text="与当前资料无关的旧文档内容。", source="old-project.pdf")
    calls = {"count": 0}

    class FakeRemote:
        def search(self, query, entities, limit=20):
            calls["count"] += 1
            return [
                RetrievalResult(chunk=stale, score=0.99, rank=1),
                RetrievalResult(chunk=active, score=0.80, rank=2),
            ]

    retriever = MilvusHybridRetriever(
        FakeRemote(),
        [active],
        embedding=HashEmbeddingAdapter(config),
        config=config,
    )

    results = retriever.search("函数极限", EntityExtractionResult(raw_query="函数极限"), limit=5)

    assert results
    assert all(result.chunk.id == "active" for result in results)

    unsupported = retriever.search("奖学金制度", EntityExtractionResult(raw_query="奖学金制度"), limit=5)
    unsupported_short_term = retriever.search("事务锁", EntityExtractionResult(raw_query="事务锁"), limit=5)

    assert unsupported
    assert unsupported_short_term
    assert filter_unanchored_candidates(
        "奖学金制度",
        EntityExtractionResult(raw_query="奖学金制度"),
        unsupported,
        active_chunks=[active],
    ) == []
    assert filter_unanchored_candidates(
        "事务锁",
        EntityExtractionResult(raw_query="事务锁"),
        unsupported_short_term,
        active_chunks=[active],
    ) == []
    assert calls["count"] == 2


def test_milvus_retriever_caches_remote_context_for_next_query():
    config = offline_config()
    active = DocumentChunk(id="active", text="函数极限描述函数值的变化趋势。", source="current.pdf")
    remote_chunk = DocumentChunk(id="remote", text="奖学金制度设有五类奖学金。", source="handbook.pdf")
    calls = {"count": 0}

    class FakeRemote:
        def search(self, query, entities, limit=20):
            calls["count"] += 1
            return [RetrievalResult(chunk=remote_chunk, score=0.9, rank=1)]

    retriever = MilvusHybridRetriever(
        FakeRemote(),
        [active],
        embedding=HashEmbeddingAdapter(config),
        config=config,
    )

    first = retriever.search("奖学金制度", EntityExtractionResult(raw_query="奖学金制度"), limit=5)
    assert first
    assert calls["count"] == 1

    assert retriever.cache_chunks([result.chunk for result in first]) == [remote_chunk]

    second = retriever.search("奖学金制度", EntityExtractionResult(raw_query="奖学金制度"), limit=5)
    assert second
    assert second[0].chunk.id == "remote"
    assert calls["count"] == 1


def test_context_filter_removes_toc_and_distant_one_word_mentions():
    toc = DocumentChunk(
        id="toc",
        text="学生奖学金制度",
        source="student-handbook.pdf",
        page=2,
        needs_review=True,
    )
    body = DocumentChunk(
        id="body",
        text="学生手册98 学生奖学金制度。设置5类奖学金，评奖每年进行一次。",
        source="student-handbook.pdf",
        page=98,
    )
    distant_mention = DocumentChunk(
        id="distant",
        text="生活委员负责管理发放奖学金。",
        source="student-handbook.pdf",
        page=164,
    )

    filtered = filter_low_quality_contexts(
        "奖学金制度是什么",
        [
            RetrievalResult(toc, 0.95),
            RetrievalResult(distant_mention, 0.80),
            RetrievalResult(body, 0.75),
        ],
    )

    assert [result.chunk.id for result in filtered] == ["body"]


def test_definition_answer_keeps_policy_purpose_and_categories():
    config = offline_config(max_answer_chars_per_source=420)
    chunk = DocumentChunk(
        id="scholarship",
        text=(
            "学生奖学金制度。第一条为了鼓励学生奋发向上，制定本制度。"
            "第四条设置学业优等生奖学金、优秀学生干部奖学金，共计5类奖学金。"
        ),
        source="student-handbook.pdf",
        page=98,
    )

    answer = RuleBasedLLMRouter(config).generate_answer(
        "奖学金制度是什么",
        [RetrievalResult(chunk, 0.9)],
    )

    assert "制定本制度" in answer
    assert "共计5类奖学金" in answer
    assert "来源" not in answer
    assert "student-handbook.pdf" not in answer


def test_hallucination_check_accepts_page_citations_from_retrieved_metadata():
    config = offline_config()
    chunk = DocumentChunk(id="handbook", text="学生奖学金制度共计5类奖学金。", source="handbook.pdf", page=98)
    source_texts = source_texts_for_validation([RetrievalResult(chunk, 0.9)])

    result = RuleBasedLLMRouter(config).hallucination_check(
        "学生手册第98页说明该制度共计5类奖学金。",
        source_texts,
    )

    assert result.passed is True


def test_short_subject_term_is_not_misclassified_as_chit_chat():
    result = RuleBasedQueryProcessor().classify_intent("事务锁")

    assert result.category == "document_answer"
    assert result.allowed is True


def test_unanchored_remote_candidates_are_rejected_for_out_of_kb_query():
    active = DocumentChunk(id="active", text="学生奖学金制度共计5类奖学金。", source="current.pdf")
    remote_near_miss = DocumentChunk(id="remote", text="学生社团管理办法。", source="current.pdf")

    filtered = filter_unanchored_candidates(
        "事务锁",
        EntityExtractionResult(raw_query="事务锁"),
        [RetrievalResult(remote_near_miss, 0.9)],
        active_chunks=[active],
    )

    assert filtered == []


def test_query_difficulty_routes_long_multi_part_question_to_complex():
    question = "请结合第一章和第二章，分别比较两个定义，并说明它们的适用条件、推导思路以及常见错误，最后给出复习建议。"

    assert assess_query_difficulty(question) == "complex"


def test_sse_answer_endpoint_streams_metadata_tokens_and_done():
    chunk = DocumentChunk(id="limit", text="极限定义中函数值无限接近确定常数。", source="math.pdf", page=1)
    client = TestClient(create_app(initial_chunks=[chunk], config=offline_config()))

    response = client.post("/answer/stream", json={"question": "极限定义是什么？", "session_id": "test-session"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: metadata" in response.text
    assert "event: token" in response.text
    assert response.text.count("event: token") >= 2
    assert "event: done" in response.text


def test_plain_text_multimodal_document_endpoint_runs_hierarchical_etl():
    client = TestClient(create_app(initial_chunks=[], config=offline_config()))
    content = "线性代数课程资料：矩阵可逆当且仅当行列式不为零。".encode("utf-8")

    response = client.post(
        "/kb/document",
        json={"filename": "linear-algebra.txt", "content_base64": base64.b64encode(content).decode("ascii")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["parser_chain"] == ["plain-text"]
    assert payload["summary"]["parent_count"] == 1
