import json

from manufacturing_rag.adapters.llm import RuleBasedLLMRouter
from manufacturing_rag.config import RAGConfig
from manufacturing_rag.pipeline import ManufacturingRAGPipeline
from manufacturing_rag.schemas import DocumentChunk


def agent_config(**overrides):
    values = {
        "llm_provider": "rule",
        "embedding_provider": "hash",
        "rerank_provider": "rule",
        "vector_store_provider": "memory",
        "query_classifier_provider": "rule",
        "query_agent_provider": "langchain",
        "embedding_dimensions": 64,
        "debug_log_enabled": False,
        "redis_enabled": False,
        "mysql_enabled": False,
        "agent_loop_enabled": True,
        "agent_request_timeout_seconds": 10,
    }
    values.update(overrides)
    return RAGConfig(**values)


def tool_call(name, arguments, call_id):
    return {
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }],
    }


FIVE_A_TEXT = (
    "本校5A人才指积极进取（Aggressive）、适应性强（Adaptable）、勤勉敬业（Assiduous）、"
    "善于学习（Academic）、精技术技能（Able）。"
)


class ScriptedAgentRouter(RuleBasedLLMRouter):
    agent_system_prompt = "测试主 Agent"

    def __init__(self, config, decisions):
        super().__init__(config)
        self.decisions = list(decisions)
        self.messages = []

    def decide_agent_action(self, messages, tools, timeout_seconds=None):
        self.messages.append([dict(message) for message in messages])
        return self.decisions.pop(0)


def make_pipeline(router, chunks=None):
    config = router.config
    return ManufacturingRAGPipeline(
        chunks or [DocumentChunk(id="five-a", text=FIVE_A_TEXT, source="学生手册.pdf", page=18)],
        config=config,
        llm_router=router,
    )


def test_main_agent_searches_then_reads_and_answers_from_selected_tools():
    config = agent_config()
    router = ScriptedAgentRouter(config, [])

    def decide(messages, tools, timeout_seconds=None):
        if not any(message.get("role") == "tool" for message in messages):
            return tool_call("search_knowledge", {"query": "5A人才是什么？"}, "call-search")
        previous_calls = messages[-2].get("tool_calls") if len(messages) >= 2 else None
        previous_name = (
            previous_calls[0].get("function", {}).get("name")
            if previous_calls and isinstance(previous_calls[0], dict)
            else ""
        )
        if previous_name == "search_knowledge":
            search_result = json.loads(messages[-1]["content"])
            evidence_id = search_result["data"]["evidence"][0]["evidence_id"]
            return tool_call("read_source", {"evidence_id": evidence_id}, "call-read")
        return {"content": "本校5A人才强调积极进取、适应性强、勤勉敬业、善于学习和精技术能。"}

    router.decide_agent_action = decide
    pipeline = make_pipeline(router)
    tokens = []
    statuses = []

    response = pipeline.answer("5A人才是什么？", on_token=tokens.append, on_status=statuses.append)

    assert "勤勉敬业" in response.answer
    assert response.sources[0].source_path == "学生手册.pdf"
    assert response.hallucination_check.passed
    assert tokens == [response.answer]
    assert "正在查询知识库…" in statuses
    assert "正在读取资料原文…" in statuses


def test_direct_chat_does_not_force_classification_or_retrieval():
    config = agent_config()
    router = ScriptedAgentRouter(config, [{"content": "你好！有什么我可以帮你的？"}])
    pipeline = make_pipeline(router)
    pipeline.query_optimizer.classify_intent = lambda *_args: (_ for _ in ()).throw(AssertionError("旧路由不应执行"))

    response = pipeline.answer("你好")

    assert response.answer.startswith("你好")
    assert response.intent.category == "chit_chat"
    assert response.sources == []
    assert len(router.messages) == 1
    assert not router.messages[0][1].get("tool_calls")


def test_main_agent_can_directly_answer_product_discussion_without_local_intent_gate():
    config = agent_config()
    router = ScriptedAgentRouter(config, [{
        "content": json.dumps(
            {"response_type": "conversation", "answer": "删除操作不需要调用问答模型；之前是界面共用了问答加载状态。"},
            ensure_ascii=False,
        ),
    }])
    pipeline = make_pipeline(router)

    response = pipeline.answer("为什么删除时 AI 会显示正在思考？")

    assert "问答加载状态" in response.answer
    assert response.sources == []
    assert "普通交互由主 Agent 直接回复" in response.notices
    assert len(router.messages) == 1


def test_main_agent_must_search_before_publishing_a_knowledge_answer():
    config = agent_config()
    router = ScriptedAgentRouter(config, [
        {"content": json.dumps({"response_type": "knowledge", "answer": "5A人才由五项素质组成。"}, ensure_ascii=False)},
        tool_call("search_knowledge", {"query": "5A人才是什么？"}, "call-search"),
        {"content": json.dumps({"response_type": "knowledge", "answer": FIVE_A_TEXT}, ensure_ascii=False)},
    ])
    pipeline = make_pipeline(router)

    response = pipeline.answer("5A人才是什么？")

    assert response.sources
    assert response.hallucination_check.passed
    assert len(router.messages) == 3
    feedback = json.loads(router.messages[1][-1]["content"])
    assert "必须先调用 search_knowledge" in feedback["reason"]


def test_failed_publication_returns_to_main_agent_and_only_validated_text_streams():
    config = agent_config()
    router = ScriptedAgentRouter(config, [
        tool_call("search_knowledge", {"query": "5A人才是什么？"}, "call-search"),
        {"content": "5A人才每项占20分，总分100分。"},
        {"content": "本校5A人才强调积极进取、适应性强、勤勉敬业、善于学习和精技术能。"},
    ])
    pipeline = make_pipeline(router)
    tokens = []

    response = pipeline.answer("5A人才是什么？", on_token=tokens.append)

    assert response.hallucination_check.passed
    assert "20分" not in response.answer
    assert tokens == [response.answer]
    feedback = json.loads(router.messages[2][-1]["content"])
    assert feedback["runtime_observation"] == "发布校验未通过。正文尚未发送给用户。"
    assert "资料未支撑" in feedback["reason"]


def test_main_agent_can_suspend_for_required_user_information():
    config = agent_config()
    router = ScriptedAgentRouter(config, [
        tool_call("collect_user_info", {"question": "你想了解哪类奖学金？"}, "call-ask"),
    ])
    pipeline = make_pipeline(router)

    response = pipeline.answer("奖学金标准")

    assert response.answer == "你想了解哪类奖学金？"
    assert "主 Agent 等待用户补充信息" in response.notices
    assert response.sources == []
    assert len(router.messages) == 1
