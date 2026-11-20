"""Read-only campus knowledge capabilities exposed to the main Agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from manufacturing_rag.agent_runtime import AgentTool, ToolResult, error_result
from manufacturing_rag.schemas import RetrievalResult


@dataclass
class KnowledgeToolState:
    """Evidence and search outcomes collected during one Agent request."""

    evidence: Dict[str, RetrievalResult] = field(default_factory=dict)
    search_queries: List[str] = field(default_factory=list)
    search_statuses: List[str] = field(default_factory=list)


def build_knowledge_tools(pipeline: Any, rerank_rules: Dict[str, Any] | None = None):
    """Create tools bound to the already authorized workspace pipeline."""

    state = KnowledgeToolState()

    def search_knowledge(arguments: Dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return error_result("EMPTY_QUERY", "检索语句不能为空。", retryable=False)
        state.search_queries.append(query)
        try:
            results = pipeline.search_knowledge(query, rerank_rules=rerank_rules)
        except Exception:
            state.search_statuses.append("error")
            raise
        if not results:
            state.search_statuses.append("empty")
            return ToolResult(
                status="empty",
                data={"query": query, "evidence": []},
                meta={"has_more": False, "new_evidence_count": 0},
            )

        evidence = []
        new_ids = set()
        for result in results:
            evidence_id = str(result.chunk.id)
            if evidence_id not in state.evidence:
                new_ids.add(evidence_id)
            previous = state.evidence.get(evidence_id)
            if previous is None or result.score > previous.score:
                state.evidence[evidence_id] = result
            chunk = result.chunk
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "source_path": chunk.source,
                    "page": chunk.page,
                    "block_id": chunk.block_id,
                    "score": round(float(result.score), 4),
                    "text": chunk.text[:2400],
                }
            )
        state.search_statuses.append("ok")
        return ToolResult(
            status="ok",
            data={"query": query, "evidence": evidence},
            meta={"has_more": False, "new_evidence_count": len(new_ids)},
        )

    def read_source(arguments: Dict[str, Any]) -> ToolResult:
        evidence_id = str(arguments.get("evidence_id") or "").strip()
        result = state.evidence.get(evidence_id)
        if result is None:
            return error_result(
                "EVIDENCE_NOT_FOUND",
                "只能读取本轮检索已返回的资料 ID；请先查询知识库或选择已返回的 ID。",
                retryable=True,
            )
        chunk = result.chunk
        text = str(chunk.metadata.get("parent_text") or chunk.text)
        return ToolResult(
            status="ok",
            data={
                "evidence_id": evidence_id,
                "source_path": chunk.source,
                "page": chunk.page,
                "block_id": chunk.block_id,
                "text": text[:8000],
            },
            meta={"truncated": len(text) > 8000},
        )

    def collect_user_info(arguments: Dict[str, Any]) -> ToolResult:
        question = str(arguments.get("question") or "").strip()
        if not question:
            return error_result("EMPTY_USER_QUESTION", "补充问题不能为空。", retryable=False)
        if re.search(r"密码|口令|密钥|api[_ -]?key|token|验证码|身份证号|银行卡号", question, re.I):
            return error_result("SENSITIVE_USER_QUESTION", "不能向用户索要凭据或高度敏感个人信息。", retryable=False)
        safety_guard = getattr(pipeline, "safety_guard", None)
        assess = getattr(safety_guard, "assess_question", None)
        safety = assess(question) if callable(assess) else None
        if safety is not None and not safety.allowed:
            return error_result("UNSAFE_USER_QUESTION", "补充问题触发安全规则，不能发送给用户。", retryable=False)
        return ToolResult(
            status="await_user",
            data={"question": question},
            meta={"suspends_request": True},
        )

    shared_object_schema = {
        "type": "object",
        "additionalProperties": False,
    }
    return [
        AgentTool(
            name="search_knowledge",
            description=(
                "检索当前请求已授权的校园知识库。查询语句由你根据用户目标决定；"
                "结果可能为空，也可能只提供部分事实。不会跨越当前工作区权限。"
            ),
            parameters={
                **shared_object_schema,
                "properties": {
                    "query": {"type": "string", "description": "本次希望检索的具体问题或关键词", "maxLength": 2000},
                },
                "required": ["query"],
            },
            handler=search_knowledge,
        ),
        AgentTool(
            name="read_source",
            description="读取本轮 search_knowledge 已返回的资料原文或所在父段落。",
            parameters={
                **shared_object_schema,
                "properties": {
                    "evidence_id": {"type": "string", "description": "此前检索结果中的 evidence_id", "maxLength": 256},
                },
                "required": ["evidence_id"],
            },
            handler=read_source,
        ),
        AgentTool(
            name="collect_user_info",
            description=(
                "只有用户必须提供、且无法通过已授权资料查明的关键条件缺失时使用。"
                "该动作会把问题返回给用户并暂停本次处理。"
            ),
            parameters={
                **shared_object_schema,
                "properties": {
                    "question": {"type": "string", "description": "向用户提出的一个简短、具体问题", "maxLength": 500},
                },
                "required": ["question"],
            },
            handler=collect_user_info,
        ),
    ], state
