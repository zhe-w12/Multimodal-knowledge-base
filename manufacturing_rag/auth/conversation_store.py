from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError

from manufacturing_rag.auth.models import Conversation, ConversationMessage, User, utcnow
from manufacturing_rag.auth.store import AuthServiceUnavailable, AuthStore


class ConversationNotFound(RuntimeError):
    pass


class ConversationLocked(RuntimeError):
    pass


class ConversationStore:
    """Private conversation history tied to the server-authenticated user."""

    def __init__(self, auth: AuthStore):
        self.auth = auth

    @staticmethod
    def _sources_json(sources: Iterable[dict[str, Any]] | None) -> str:
        safe_sources = []
        for source in sources or []:
            if not isinstance(source, dict):
                continue
            safe_sources.append(source)
        return json.dumps(safe_sources, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _message_dict(message: ConversationMessage) -> dict[str, Any]:
        try:
            sources = json.loads(message.sources_json or "[]")
        except (TypeError, json.JSONDecodeError):
            sources = []
        return {
            "id": message.id,
            "role": message.role,
            "text": message.content,
            "sources": sources if isinstance(sources, list) else [],
            "created_at": message.created_at.isoformat(),
        }

    @staticmethod
    def _conversation_dict(conversation: Conversation, *, include_messages: bool = False, messages=None) -> dict[str, Any]:
        result = {
            "id": conversation.id,
            "scope_type": conversation.scope_type,
            "workspace_id": conversation.workspace_id,
            "title": conversation.title,
            "status": conversation.status,
            "locked_reason": conversation.locked_reason,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
        }
        if include_messages:
            result["messages"] = [ConversationStore._message_dict(message) for message in (messages or [])]
        return result

    def _owned(self, db, user_id: str, conversation_id: str, *, lock: bool = False) -> Conversation:
        statement = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        if lock:
            statement = statement.with_for_update()
        conversation = db.scalar(statement)
        if conversation is None:
            raise ConversationNotFound("对话不存在")
        return conversation

    def create(self, user_id: str, scope_type: str, workspace_id: str | None = None) -> dict[str, Any]:
        if scope_type not in {"all", "common", "workspace"}:
            raise ValueError("对话范围无效")
        if scope_type == "workspace" and not workspace_id:
            raise ValueError("请选择工作区")
        if scope_type != "workspace":
            workspace_id = None
        self.auth.ensure_schema()
        try:
            with self.auth.Session.begin() as db:
                user = db.get(User, user_id)
                if user is None or user.status != "active":
                    raise ConversationNotFound("账号不可用")
                conversation = Conversation(
                    id=str(uuid4()),
                    user_id=user_id,
                    scope_type=scope_type,
                    workspace_id=workspace_id,
                    title="新对话",
                    status="active",
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
                db.add(conversation)
                db.flush()
                return self._conversation_dict(conversation)
        except (ConversationNotFound, ValueError):
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("对话数据服务暂不可用") from exc

    def list(self, user_id: str) -> list[dict[str, Any]]:
        self.auth.ensure_schema()
        try:
            with self.auth.Session() as db:
                message_count = (
                    select(func.count(ConversationMessage.id))
                    .where(ConversationMessage.conversation_id == Conversation.id)
                    .correlate(Conversation)
                    .scalar_subquery()
                )
                rows = db.execute(
                    select(Conversation, message_count)
                    .where(Conversation.user_id == user_id)
                    .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                ).all()
                return [
                    {**self._conversation_dict(conversation), "message_count": int(count or 0)}
                    for conversation, count in rows
                ]
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("对话数据服务暂不可用") from exc

    def detail(self, user_id: str, conversation_id: str, readable_workspace_ids: Iterable[str]) -> dict[str, Any]:
        self.auth.ensure_schema()
        readable = set(readable_workspace_ids)
        try:
            with self.auth.Session.begin() as db:
                conversation = self._owned(db, user_id, conversation_id, lock=True)
                messages = db.scalars(
                    select(ConversationMessage)
                    .where(ConversationMessage.conversation_id == conversation.id)
                    .order_by(ConversationMessage.sequence)
                ).all()
                if conversation.status == "active":
                    source_workspace_ids: set[str] = set()
                    for message in messages:
                        try:
                            sources = json.loads(message.sources_json or "[]")
                        except (TypeError, json.JSONDecodeError):
                            sources = []
                        for source in sources if isinstance(sources, list) else []:
                            metadata = source.get("metadata") if isinstance(source, dict) else {}
                            source_workspace_ids.add(str((metadata or {}).get("workspace_id") or "campus-default"))
                    scope_lost = (
                        conversation.scope_type == "workspace" and conversation.workspace_id not in readable
                    ) or bool(source_workspace_ids - readable)
                    if scope_lost:
                        conversation.status = "locked"
                        conversation.locked_reason = "资料权限已变更"
                        conversation.updated_at = utcnow()
                if conversation.status == "locked":
                    return self._conversation_dict(conversation, include_messages=True, messages=[])
                return self._conversation_dict(conversation, include_messages=True, messages=messages)
        except ConversationNotFound:
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("对话数据服务暂不可用") from exc

    def begin_turn(
        self,
        user_id: str,
        conversation_id: str,
        scope_type: str,
        workspace_id: str | None,
        readable_workspace_ids: Iterable[str],
        question: str,
    ) -> str | None:
        self.auth.ensure_schema()
        readable = set(readable_workspace_ids)
        try:
            with self.auth.Session.begin() as db:
                conversation = self._owned(db, user_id, conversation_id, lock=True)
                if conversation.status != "active":
                    raise ConversationLocked("资料权限已变更，请新建对话")
                if conversation.scope_type != scope_type or (
                    scope_type == "workspace" and conversation.workspace_id != workspace_id
                ):
                    raise ValueError("当前对话的检索范围已固定，请新建对话")
                if scope_type == "workspace" and workspace_id not in readable:
                    conversation.status = "locked"
                    conversation.locked_reason = "资料权限已变更"
                    conversation.updated_at = utcnow()
                    return None
                messages = db.scalars(
                    select(ConversationMessage)
                    .where(ConversationMessage.conversation_id == conversation.id)
                    .order_by(ConversationMessage.sequence)
                ).all()
                source_workspace_ids: set[str] = set()
                for message in messages:
                    try:
                        sources = json.loads(message.sources_json or "[]")
                    except (TypeError, json.JSONDecodeError):
                        sources = []
                    for source in sources if isinstance(sources, list) else []:
                        metadata = source.get("metadata") if isinstance(source, dict) else {}
                        source_workspace_ids.add(str((metadata or {}).get("workspace_id") or "campus-default"))
                if source_workspace_ids - readable:
                    conversation.status = "locked"
                    conversation.locked_reason = "资料权限已变更"
                    conversation.updated_at = utcnow()
                    return None
                context_rows = messages[-8:]
                context = "\n".join(
                    f"{'用户' if row.role == 'user' else '助手'}：{row.content}"
                    for row in context_rows
                    if row.content.strip()
                )[-4000:]
                sequence = (messages[-1].sequence + 1) if messages else 1
                db.add(
                    ConversationMessage(
                        id=str(uuid4()),
                        conversation_id=conversation.id,
                        sequence=sequence,
                        role="user",
                        content=question,
                        sources_json="[]",
                        created_at=utcnow(),
                    )
                )
                if not messages and conversation.title == "新对话":
                    conversation.title = question.strip()[:30] or "新对话"
                conversation.updated_at = utcnow()
                return context
        except (ConversationNotFound, ConversationLocked, ValueError):
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("对话数据服务暂不可用") from exc

    def append_assistant(
        self,
        user_id: str,
        conversation_id: str,
        answer: str,
        sources: Iterable[dict[str, Any]],
    ) -> None:
        self.auth.ensure_schema()
        try:
            with self.auth.Session.begin() as db:
                conversation = self._owned(db, user_id, conversation_id, lock=True)
                if conversation.status != "active":
                    raise ConversationLocked("资料权限已变更")
                latest = db.scalar(
                    select(func.max(ConversationMessage.sequence)).where(
                        ConversationMessage.conversation_id == conversation.id
                    )
                ) or 0
                db.add(
                    ConversationMessage(
                        id=str(uuid4()),
                        conversation_id=conversation.id,
                        sequence=int(latest) + 1,
                        role="assistant",
                        content=answer,
                        sources_json=self._sources_json(sources),
                        created_at=utcnow(),
                    )
                )
                conversation.updated_at = utcnow()
        except (ConversationNotFound, ConversationLocked):
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("对话数据服务暂不可用") from exc

    def rename(self, user_id: str, conversation_id: str, title: str) -> dict[str, Any]:
        title = title.strip()
        if not title or len(title) > 120:
            raise ValueError("标题长度须为 1～120 个字符")
        self.auth.ensure_schema()
        try:
            with self.auth.Session.begin() as db:
                conversation = self._owned(db, user_id, conversation_id, lock=True)
                conversation.title = title
                conversation.updated_at = utcnow()
                return self._conversation_dict(conversation)
        except (ConversationNotFound, ValueError):
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("对话数据服务暂不可用") from exc

    def delete(self, user_id: str, conversation_id: str) -> None:
        self.auth.ensure_schema()
        try:
            with self.auth.Session.begin() as db:
                conversation = self._owned(db, user_id, conversation_id, lock=True)
                db.execute(delete(Conversation).where(Conversation.id == conversation.id))
        except ConversationNotFound:
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("对话数据服务暂不可用") from exc
