import json

import pytest

from manufacturing_rag.agent_runtime import (
    AgentExecutionInterrupted,
    AgentRuntime,
    AgentTool,
    FinalValidation,
    ToolRegistry,
    ToolResult,
)


def call(name, arguments, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class ScriptedModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.messages = []

    def decide_agent_action(self, messages, tools, timeout_seconds=None):
        self.messages.append([dict(message) for message in messages])
        return self.decisions.pop(0)


def simple_tool(name, handler):
    return AgentTool(
        name=name,
        description="test tool",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "maxLength": 20}},
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def test_model_controls_tool_order_and_receives_results_each_turn():
    observations = []
    tools = ToolRegistry([
        simple_tool("search_knowledge", lambda args: ToolResult("ok", {"evidence_id": "doc-1"})),
        AgentTool(
            name="read_source",
            description="read source",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda args: ToolResult("ok", {"text": "正文"}),
        ),
    ])
    model = ScriptedModel([
        {"tool_calls": [call("search_knowledge", {"query": "先搜"})], "reasoning_content": "private reasoning"},
        {"tool_calls": [call("read_source", {"query": "doc-1"}, "call-2")]},
        {"content": "完成"},
    ])

    result = AgentRuntime(model, tools, "system", 10).run(
        "问题",
        "",
        validate_final=lambda answer, state: FinalValidation(True, answer=answer),
    )

    assert result.answer == "完成"
    assert len(model.messages) == 3
    assert model.messages[1][-2]["reasoning_content"] == "private reasoning"
    assert json.loads(model.messages[1][-1]["content"])["data"]["evidence_id"] == "doc-1"
    assert json.loads(model.messages[2][-1]["content"])["data"]["text"] == "正文"
    assert [row["tool_name"] for row in result.state.observations] == ["search_knowledge", "read_source"]


def test_tool_error_is_an_observation_and_agent_can_continue():
    model = ScriptedModel([
        {"tool_calls": [call("search_knowledge", {"query": "失败"})]},
        {"content": "我无法确认。"},
    ])
    tools = ToolRegistry([
        simple_tool(
            "search_knowledge",
            lambda _args: ToolResult("error", error={"code": "VECTOR_UNAVAILABLE", "message": "检索服务暂不可用", "retryable": True}),
        ),
    ])

    result = AgentRuntime(model, tools, "system", 10).run(
        "问题", "", validate_final=lambda answer, state: FinalValidation(True, answer=answer)
    )

    assert result.answer == "我无法确认。"
    observation = json.loads(model.messages[1][-1]["content"])
    assert observation["status"] == "error"
    assert observation["error"]["code"] == "VECTOR_UNAVAILABLE"


def test_agent_can_repeat_actions_without_business_round_limit():
    decisions = [
        {"tool_calls": [call("search_knowledge", {"query": "同一问题"}, f"call-{index}")]}
        for index in range(5)
    ] + [{"content": "最终回复"}]
    model = ScriptedModel(decisions)
    tools = ToolRegistry([
        simple_tool("search_knowledge", lambda _args: ToolResult("empty", {"evidence": []})),
    ])

    result = AgentRuntime(model, tools, "system", 10).run(
        "问题", "", validate_final=lambda answer, state: FinalValidation(True, answer=answer)
    )

    assert result.answer == "最终回复"
    assert len(model.messages) == 6
    repeated = json.loads(model.messages[-1][-1]["content"])
    assert repeated["meta"]["repeat_count"] == 5


def test_collect_user_info_suspends_and_must_be_exclusive():
    execution = {"search": 0}
    tools = ToolRegistry([
        AgentTool(
            "collect_user_info",
            "ask",
            {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"], "additionalProperties": False},
            lambda args: ToolResult("await_user", {"question": args["question"]}),
        ),
        simple_tool("search_knowledge", lambda _args: execution.__setitem__("search", execution["search"] + 1) or ToolResult("ok")),
    ])
    model = ScriptedModel([
        {"tool_calls": [
            call("collect_user_info", {"question": "你想了解哪一类？"}, "call-ask"),
            call("search_knowledge", {"query": "奖学金"}, "call-search"),
        ]},
        {"content": "等待补充"},
    ])

    result = AgentRuntime(model, tools, "system", 10).run(
        "问题", "", validate_final=lambda answer, state: FinalValidation(True, answer=answer)
    )

    assert execution["search"] == 0
    assert result.answer == "等待补充"
    assert model.messages[1][-1]["role"] == "tool"
    assert json.loads(model.messages[1][-1]["content"])["error"]["code"] == "AWAIT_USER_MUST_BE_EXCLUSIVE"


def test_single_user_info_tool_returns_without_another_model_turn():
    model = ScriptedModel([{"tool_calls": [call("collect_user_info", {"question": "请补充专业名称。"})]}])
    tools = ToolRegistry([
        AgentTool(
            "collect_user_info",
            "ask",
            {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"], "additionalProperties": False},
            lambda args: ToolResult("await_user", {"question": args["question"]}),
        ),
    ])

    result = AgentRuntime(model, tools, "system", 10).run(
        "问题", "", validate_final=lambda answer, state: pytest.fail("不应进入最终回复校验")
    )

    assert result.answer == "请补充专业名称。"
    assert result.awaiting_user is True
    assert len(model.messages) == 1


def test_validation_feedback_returns_to_model_before_answer_is_accepted():
    model = ScriptedModel([{"content": "未核验草稿"}, {"content": "最终答案"}])

    result = AgentRuntime(model, ToolRegistry([]), "system", 10).run(
        "问题",
        "",
        validate_final=lambda answer, state: (
            FinalValidation(False, feedback="需先查询资料") if answer == "未核验草稿" else FinalValidation(True, answer=answer)
        ),
    )

    assert result.answer == "最终答案"
    assert json.loads(model.messages[1][-1]["content"])["runtime_observation"] == "发布校验未通过。正文尚未发送给用户。"


def test_tool_schema_rejects_extra_and_wrongly_typed_arguments():
    registry = ToolRegistry([simple_tool("search_knowledge", lambda args: ToolResult("ok", args))])

    assert registry.execute("search_knowledge", {"query": "x", "workspace_id": "other"}).error["code"] == "INVALID_ARGUMENTS"
    assert registry.execute("search_knowledge", {"query": 4}).error["code"] == "INVALID_ARGUMENTS"
    assert registry.execute("search_knowledge", {"query": "x" * 21}).error["code"] == "INVALID_ARGUMENTS"


def test_cancellation_is_an_infrastructure_interrupt():
    model = ScriptedModel([{"content": "should not run"}])
    with pytest.raises(AgentExecutionInterrupted) as error:
        AgentRuntime(model, ToolRegistry([]), "system", 10).run(
            "问题", "", validate_final=lambda answer, state: FinalValidation(True), cancel_check=lambda: True
        )
    assert error.value.code == "cancelled"
    assert not model.messages
