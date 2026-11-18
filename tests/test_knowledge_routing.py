import pytest

from manufacturing_rag.adapters.llm import RuleBasedLLMRouter
from manufacturing_rag.config import RAGConfig
from manufacturing_rag.pipeline import ManufacturingRAGPipeline
from manufacturing_rag.query import split_terms
from manufacturing_rag.schemas import DocumentChunk, IntentResult


def pipeline_with_chat_misclassification():
    config = RAGConfig(
        llm_provider="rule", embedding_provider="hash", rerank_provider="rule",
        vector_store_provider="memory", query_classifier_provider="rule", query_agent_provider="rule",
        embedding_dimensions=64, debug_log_enabled=False, redis_enabled=False, mysql_enabled=False,
    )

    class Router(RuleBasedLLMRouter):
        def classify_intent(self, question, conversation_context=""):
            return IntentResult("chit_chat", True, .99, "模拟模型将概念问题误分为闲聊")

        def generate_chat_response(self, question, conversation_context=""):
            return "普通闲聊回复"

    return ManufacturingRAGPipeline([
        DocumentChunk(id="sample", text="合成测试资料：5A 人才培养注重学生的素质、知识与能力协调发展。", source="测试手册.pdf"),
    ], config=config, llm_router=Router(config))


@pytest.mark.parametrize("question", ["5A是什么?", "5A人才是什么?", "5a人才是什么？"])
@pytest.mark.parametrize("stream", [False, True])
def test_concept_queries_retrieve_sources_even_when_model_classifies_them_as_chat(question, stream):
    pipeline = pipeline_with_chat_misclassification()
    tokens = []
    response = pipeline.answer(question, on_token=tokens.append if stream else None)
    assert response.intent.category != "chit_chat"
    assert response.sources
    assert "协调发展" in response.answer
    assert "普通闲聊" not in response.answer
    if stream:
        assert "".join(tokens) == response.answer


@pytest.mark.parametrize("question", ["你好", "谢谢", "你是谁", "你能做什么", "今天天气怎么样"])
def test_social_messages_and_assistant_questions_keep_chat_route(question):
    response = pipeline_with_chat_misclassification().answer(question)
    assert response.intent.category == "chit_chat"
    assert not response.sources


def test_blocked_intent_is_not_rerouted_to_retrieval():
    pipeline = pipeline_with_chat_misclassification()
    pipeline.query_optimizer.classify_intent = lambda *args: IntentResult("unauthorized_query", False, 1, "禁止访问")
    assert not pipeline.answer("密码是什么").intent.allowed


def test_mixed_alphanumeric_terms_remain_searchable():
    terms = split_terms("5A 人才、3D 技术和 A5 资料，标准为80分")
    assert {"5a", "3d", "a5", "80分"}.issubset(terms)
