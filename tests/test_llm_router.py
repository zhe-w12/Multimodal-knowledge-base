import json

from manufacturing_rag.adapters.llm import (
    DeepSeekClient,
    DeepSeekLLMRouter,
    RuleBasedLLMRouter,
    answer_is_grounded_by_sources,
    as_bool,
    build_llm_router,
    extract_json_object,
)
from manufacturing_rag.config import RAGConfig
from manufacturing_rag.schemas import DocumentChunk, RetrievalResult


def test_build_llm_router_defaults_to_rule_provider():
    router = build_llm_router(RAGConfig(llm_provider="rule"))

    assert isinstance(router, RuleBasedLLMRouter)


def test_extract_json_object_accepts_code_fenced_json():
    payload = extract_json_object('```json\n{"passed": true, "issues": []}\n```')

    assert payload == {"passed": True, "issues": []}


def test_as_bool_parses_string_values():
    assert as_bool("false", True) is False
    assert as_bool("通过", False) is True
    assert as_bool(None, True) is True


def test_answer_grounded_by_sources_accepts_mysql_lock_answer():
    answer = "乐观锁不加真实锁，通过版本号或时间戳控制并发，适合读多写少场景。"
    source = "①乐观锁(Optimistic Lock)\n-不加真实锁，通过版本号/时间戳控制并发\n-适合读多写少场景，无锁竞争、性能高"

    assert answer_is_grounded_by_sources(answer, [source]) is True


def test_deepseek_router_sends_simple_answer_to_configured_llm(monkeypatch):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "test-key")
    config = RAGConfig(
        llm_provider="deepseek",
        llm_api_key_env="TEST_DEEPSEEK_KEY",
        embedding_provider="hash",
        rerank_provider="rule",
        vector_store_provider="memory",
        embedding_dimensions=64,
    )
    router = DeepSeekLLMRouter(config)
    calls = {"count": 0}

    class FakeClient:
        def chat(self, **kwargs):
            calls["count"] += 1
            return "奖学金制度用于鼓励学生发展。"

    router.client = FakeClient()
    result = router.generate_answer(
        "奖学金制度",
        [RetrievalResult(DocumentChunk(id="x", text="奖学金制度用于鼓励学生发展。", source="handbook.pdf"), 0.9)],
    )

    assert calls["count"] == 1
    assert result.startswith("奖学金制度用于鼓励学生发展")


def test_deepseek_client_preserves_native_tool_call_and_provider_reasoning(monkeypatch):
    response_message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "provider state",
        "tool_calls": [{
            "id": "call-search",
            "type": "function",
            "function": {"name": "search_knowledge", "arguments": "{\"query\":\"5A\"}"},
        }],
    }
    observed = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": response_message}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = DeepSeekClient("unused-test-key", "deepseek-v4-flash", "https://example.test/chat/completions", 45)
    message = client.complete_with_tools(
        messages=[{"role": "user", "content": "问题"}],
        tools=[{"type": "function", "function": {"name": "search_knowledge"}}],
        temperature=0,
        max_tokens=100,
        enable_thinking=True,
        timeout_seconds=7,
    )

    assert message == response_message
    assert observed["payload"]["tool_choice"] == "auto"
    assert observed["payload"]["thinking"] == {"type": "enabled"}
    assert observed["timeout"] == 7
