import json

import pytest

from manufacturing_rag.adapters.llm import DeepSeekLLMRouter, RuleBasedLLMRouter
from manufacturing_rag.config import RAGConfig
from manufacturing_rag.schemas import DocumentChunk, RetrievalResult


class RecordingClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return self.response

    def chat_stream(self, **kwargs):
        self.calls.append(("chat_stream", kwargs))
        # Split inside the old insufficient-answer prefix to exercise buffering.
        for start in range(0, len(self.response), 3):
            yield self.response[start : start + 3]


def make_router(monkeypatch, response):
    monkeypatch.setenv("TEST_CLARIFICATION_API_KEY", "test-key")
    config = RAGConfig(
        llm_provider="deepseek",
        llm_api_key_env="TEST_CLARIFICATION_API_KEY",
        llm_fallback_enabled=False,
        embedding_provider="hash",
        rerank_provider="rule",
        vector_store_provider="memory",
        query_classifier_provider="rule",
        query_agent_provider="rule",
        embedding_dimensions=64,
        redis_enabled=False,
        mysql_enabled=False,
        debug_log_enabled=False,
    )
    router = DeepSeekLLMRouter(config)
    client = RecordingClient(response)
    router.client = client
    return router, client


def context_with(text):
    return [
        RetrievalResult(
            DocumentChunk(id="scholarship", text=text, source="handbook.pdf", page=98),
            0.95,
        )
    ]


def generate_answer(router, mode, question, contexts):
    if mode == "chat":
        return router.generate_answer(question, contexts)
    tokens = []
    result = router.generate_answer_stream(question, contexts, tokens.append)
    assert "".join(tokens) == result
    return result


def test_follow_up_resolution_sends_history_and_uses_independent_question(monkeypatch):
    history = (
        "用户：奖学金的奖金\n"
        "助手：你指的是哪一类奖学金？\n"
        "用户：校级优秀学生奖学金\n"
        "助手：是哪个学年、哪个学院的，例如2025学年吗？"
    )
    independent_question = "学业优等生奖学金的奖金是多少？"
    router, client = make_router(
        monkeypatch,
        json.dumps({"standalone_question": independent_question}, ensure_ascii=False),
    )

    result = router.resolve_follow_up_question("学业优等生奖学金", history)

    assert result == independent_question
    assert len(client.calls) == 1
    mode, request = client.calls[0]
    assert mode == "chat"
    sent = json.loads(request["user_prompt"])
    assert sent["task"] == "追问问题还原"
    assert sent["payload"]["question"] == "学业优等生奖学金"
    assert sent["payload"]["conversation_context"] == history


def test_resolution_that_drops_latest_award_falls_back_instead_of_using_old_category(monkeypatch):
    router, _ = make_router(
        monkeypatch,
        json.dumps({"standalone_question": "校级优秀学生奖学金的奖金是多少？"}, ensure_ascii=False),
    )
    assert router.resolve_follow_up_question(
        "学业优等生奖学金", "用户：奖学金的奖金\n用户：校级优秀学生奖学金",
    ) == ""


@pytest.mark.parametrize("mode", ["chat", "chat_stream"])
@pytest.mark.parametrize(
    "answer",
    [
        "当前资料未列出该奖项的奖金金额。",
        "当前资料不足以直接回答：未列出该奖项的奖金金额。",
    ],
)
def test_missing_amount_remains_a_direct_answer_without_another_model_call(monkeypatch, mode, answer):
    router, client = make_router(monkeypatch, answer)
    contexts = context_with("学业优等生奖学金每学年评选一次，资料未列出奖金金额。")

    result = generate_answer(router, mode, "学业优等生奖学金的奖金是多少？", contexts)

    assert result == answer
    assert len(client.calls) == 1
    assert client.calls[0][0] == mode
    assert router.last_answer_source == "remote_llm"


@pytest.mark.parametrize("mode", ["chat", "chat_stream"])
def test_supported_amount_is_answered_without_additional_clarification(monkeypatch, mode):
    answer = "学业优等生奖学金的奖金为每人1000元。"
    router, client = make_router(monkeypatch, answer)
    contexts = context_with("学业优等生奖学金的奖励标准为每人1000元。")

    sufficient, _ = router.assess_context_sufficiency("学业优等生奖学金的奖金是多少？", contexts)
    assert sufficient
    assert not client.calls

    result = generate_answer(router, mode, "学业优等生奖学金的奖金是多少？", contexts)

    assert result == answer
    assert len(client.calls) == 1
    assert client.calls[0][0] == mode
    sent = json.loads(client.calls[0][1]["user_prompt"])
    assert sent["question"] == "学业优等生奖学金的奖金是多少？"
    assert "1000元" in json.dumps(sent["contexts"], ensure_ascii=False)


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("奖金为 5000 元。", "奖金为 1000 元。"),
        ("奖金为 2 万元。", "奖金为 1 万元。"),
        ("奖金为1000元。", "奖金为21000元。"),
    ],
)
def test_amounts_not_supported_by_sources_fail_hallucination_check(answer, source):
    router = RuleBasedLLMRouter()

    result = router.hallucination_check(answer, [source])

    assert not result.passed
    assert result.issues


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("奖金为10000元。", "奖金为1万元。"),
        ("奖金为1,000元。", "奖金为1000.0元。"),
    ],
)
def test_equivalent_currency_amounts_pass_hallucination_check(answer, source):
    assert RuleBasedLLMRouter().hallucination_check(answer, [source]).passed
