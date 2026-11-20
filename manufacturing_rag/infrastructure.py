"""Redis cache and MySQL business persistence used by the FastAPI service."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Optional

from manufacturing_rag.config import DEFAULT_CONFIG, RAGConfig
from manufacturing_rag.schemas import (
    Coordinates,
    EntityExtractionResult,
    HallucinationCheck,
    IntentResult,
    RAGResponse,
    SafetyResult,
    SourceRef,
)

SESSION_MEMORY_TURNS = 8
SESSION_MEMORY_MAX_SESSIONS = 200
SESSION_MEMORY_TTL_CAP_SECONDS = 3600
SESSION_CONTEXT_MAX_CHARS = 4000


class RuntimeInfrastructure:
    """Best-effort infrastructure facade; optional outages do not break grounded answers."""

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._redis: Any = None
        self._engine: Any = None
        self._redis_error = ""
        self._mysql_error = ""
        self._session_memory: OrderedDict[str, tuple[float, list[dict[str, str]]]] = OrderedDict()
        self._session_memory_lock = RLock()

    def cache_key(self, question: str, knowledge_version: int) -> str:
        digest = hashlib.sha256(f"{knowledge_version}:{question.strip()}".encode("utf-8")).hexdigest()
        return f"education-rag:answer:v2:{digest}"

    def get_cached_answer(self, question: str, knowledge_version: int) -> Optional[RAGResponse]:
        client = self._redis_client()
        if client is None:
            return None
        try:
            raw = client.get(self.cache_key(question, knowledge_version))
            if not raw:
                return None
            return replace(response_from_dict(json.loads(raw)), cache_hit=True)
        except Exception as exc:  # noqa: BLE001 - cache outages must not block retrieval.
            self._redis_error = str(exc)
            return None

    def get_session_context(self, session_id: Optional[str], limit: int = 4) -> str:
        if not session_id:
            return ""
        client = self._redis_client()
        if client is not None:
            try:
                rows = client.lrange(
                    f"education-rag:session:{safe_session_id(session_id)}",
                    -max(1, limit),
                    -1,
                )
                payloads = [json.loads(row) for row in rows]
                context = format_session_context(payloads)
                if payloads:
                    self._remember_session_history(session_id, payloads)
                if context:
                    return context
            except Exception as exc:  # noqa: BLE001
                self._redis_error = str(exc)
        return self._get_local_session_context(session_id, limit)

    def cache_answer(
        self,
        response: RAGResponse,
        knowledge_version: int,
        session_id: Optional[str] = None,
        cache_basis: Optional[str] = None,
    ) -> None:
        if session_id:
            self._remember_session_turn(session_id, response.question, response.answer)
        client = self._redis_client()
        if client is not None:
            try:
                client.setex(
                    self.cache_key(cache_basis or response.question, knowledge_version),
                    self.config.redis_cache_ttl_seconds,
                    json.dumps(response.to_dict(), ensure_ascii=False),
                )
                if session_id:
                    session_key = f"education-rag:session:{safe_session_id(session_id)}"
                    client.rpush(
                        session_key,
                        json.dumps(
                            {"question": response.question, "answer": response.answer, "at": utc_now()},
                            ensure_ascii=False,
                        ),
                    )
                    client.ltrim(session_key, -20, -1)
                    client.expire(session_key, self.config.redis_session_ttl_seconds)
            except Exception as exc:  # noqa: BLE001
                self._redis_error = str(exc)
        self.persist_conversation(response, session_id=session_id)

    def remember_response(self, response: RAGResponse, session_id: Optional[str] = None) -> None:
        """Persist conversation context without using the legacy answer cache."""

        if session_id:
            self._remember_session_turn(session_id, response.question, response.answer)
        self.persist_conversation(response, session_id=session_id)

    def _remember_session_turn(self, session_id: str, question: str, answer: str) -> None:
        self._remember_session_history(
            session_id,
            [{"question": str(question or "")[:1000], "answer": str(answer or "")[:3000]}],
            append=True,
        )

    def _remember_session_history(
        self,
        session_id: str,
        rows: list[dict[str, Any]],
        append: bool = False,
    ) -> None:
        session_key = safe_session_id(session_id)
        now = time.monotonic()
        ttl = max(1, min(int(self.config.redis_session_ttl_seconds), SESSION_MEMORY_TTL_CAP_SECONDS))
        with self._session_memory_lock:
            self._prune_expired_session_memory(now)
            existing = self._session_memory.pop(session_key, (0.0, []))[1] if append else []
            turns = existing + [
                {
                    "question": str(row.get("question") or "")[:1000],
                    "answer": str(row.get("answer") or "")[:3000],
                }
                for row in rows
            ]
            self._session_memory[session_key] = (now + ttl, turns[-SESSION_MEMORY_TURNS:])
            self._session_memory.move_to_end(session_key)
            while len(self._session_memory) > SESSION_MEMORY_MAX_SESSIONS:
                self._session_memory.popitem(last=False)

    def _get_local_session_context(self, session_id: str, limit: int) -> str:
        session_key = safe_session_id(session_id)
        now = time.monotonic()
        with self._session_memory_lock:
            self._prune_expired_session_memory(now)
            entry = self._session_memory.get(session_key)
            if entry is None:
                return ""
            self._session_memory.move_to_end(session_key)
            rows = entry[1][-max(1, limit):]
        return format_session_context(rows)

    def _prune_expired_session_memory(self, now: float) -> None:
        expired = [key for key, (expires_at, _) in self._session_memory.items() if expires_at <= now]
        for key in expired:
            self._session_memory.pop(key, None)

    def persist_conversation(self, response: RAGResponse, session_id: Optional[str] = None) -> None:
        engine = self._mysql_engine()
        if engine is None:
            return
        try:
            from sqlalchemy import text  # type: ignore[import-not-found]

            with engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO rag_conversations
                            (session_id, question, answer, sources_json, retrieval_confidence, created_at)
                        VALUES
                            (:session_id, :question, :answer, :sources_json, :retrieval_confidence, UTC_TIMESTAMP())
                        """
                    ),
                    {
                        "session_id": safe_session_id(session_id or "anonymous"),
                        "question": response.question,
                        "answer": response.answer,
                        "sources_json": json.dumps([source.to_dict() for source in response.sources], ensure_ascii=False),
                        "retrieval_confidence": response.retrieval_confidence,
                    },
                )
        except Exception as exc:  # noqa: BLE001 - persistence is observable but not on the answer critical path.
            self._mysql_error = str(exc)

    def status(self) -> dict[str, Any]:
        redis_status = "disabled"
        if self.config.redis_enabled:
            client = self._redis_client()
            redis_status = "connected" if client is not None and not self._redis_error else "degraded"
        mysql_status = "disabled"
        if self.config.mysql_enabled:
            engine = self._mysql_engine()
            mysql_status = "connected" if engine is not None and not self._mysql_error else "degraded"
        return {
            "redis": redis_status,
            "mysql": mysql_status,
            "redis_error": bounded_error(self._redis_error),
            "mysql_error": bounded_error(self._mysql_error),
        }

    def _redis_client(self):
        if not self.config.redis_enabled:
            return None
        if self._redis is not None:
            return self._redis
        try:
            import redis  # type: ignore[import-not-found]

            client = redis.Redis.from_url(self.config.redis_url, decode_responses=True, socket_timeout=2)
            client.ping()
            self._redis = client
            self._redis_error = ""
            return client
        except Exception as exc:  # noqa: BLE001
            self._redis_error = str(exc)
            return None

    def _mysql_engine(self):
        if not self.config.mysql_enabled:
            return None
        if self._engine is not None:
            return self._engine
        try:
            from sqlalchemy import create_engine, text  # type: ignore[import-not-found]

            engine = create_engine(self.config.mysql_url, pool_pre_ping=True, pool_recycle=1800)
            with engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS rag_conversations (
                            id BIGINT AUTO_INCREMENT PRIMARY KEY,
                            session_id VARCHAR(128) NOT NULL,
                            question TEXT NOT NULL,
                            answer LONGTEXT NOT NULL,
                            sources_json JSON NOT NULL,
                            retrieval_confidence DOUBLE NOT NULL DEFAULT 0,
                            created_at DATETIME NOT NULL,
                            INDEX idx_rag_session_created (session_id, created_at)
                        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
                        """
                    )
                )
            self._engine = engine
            self._mysql_error = ""
            return engine
        except Exception as exc:  # noqa: BLE001
            self._mysql_error = str(exc)
            return None


def response_from_dict(payload: dict[str, Any]) -> RAGResponse:
    intent = payload.get("intent") or {}
    safety = payload.get("safety") or {}
    hallucination = payload.get("hallucination_check") or {}
    sources = []
    for source in payload.get("sources") or []:
        coordinates = Coordinates.from_value(source.get("coordinates"))
        sources.append(
            SourceRef(
                document_id=str(source.get("document_id") or ""),
                source_path=str(source.get("source_path") or ""),
                page=source.get("page"),
                block_id=source.get("block_id"),
                modality=str(source.get("modality") or "ocr_text"),
                coordinates=coordinates,
                score=source.get("score"),
                parent_id=source.get("parent_id"),
                chunk_level=str(source.get("chunk_level") or "child"),
                metadata=dict(source.get("metadata") or {}),
            )
        )
    return RAGResponse(
        question=str(payload.get("question") or ""),
        answer=str(payload.get("answer") or ""),
        intent=IntentResult(
            category=str(intent.get("category") or "document_answer"),
            allowed=bool(intent.get("allowed", True)),
            confidence=float(intent.get("confidence", 0.0)),
            reason=str(intent.get("reason") or ""),
            difficulty=str(intent.get("difficulty") or "simple"),
            query_strategy=str(intent.get("query_strategy") or "direct"),
        ),
        safety=SafetyResult(
            allowed=bool(safety.get("allowed", True)),
            risk_level=str(safety.get("risk_level") or "low"),
            risk_types=list(safety.get("risk_types") or []),
            reason=str(safety.get("reason") or ""),
        ),
        optimized_query=str(payload.get("optimized_query") or ""),
        query_variants=[str(item) for item in payload.get("query_variants") or []],
        entities=EntityExtractionResult(
            raw_query=str((payload.get("entities") or {}).get("raw_query") or ""),
            entities=dict((payload.get("entities") or {}).get("entities") or {}),
        ),
        sources=sources,
        retrieval_confidence=float(payload.get("retrieval_confidence") or 0.0),
        hallucination_check=HallucinationCheck(
            passed=bool(hallucination.get("passed", True)),
            issues=list(hallucination.get("issues") or []),
            confidence=float(hallucination.get("confidence", 1.0)),
        ),
        redacted=bool(payload.get("redacted", False)),
        notices=list(payload.get("notices") or []),
        cache_hit=bool(payload.get("cache_hit", False)),
    )


def safe_session_id(value: str) -> str:
    cleaned = "".join(char for char in str(value) if char.isalnum() or char in {"-", "_", "."})
    return cleaned[:128] or "anonymous"


def format_session_context(rows: list[dict[str, Any]]) -> str:
    messages = [
        f"用户：{row.get('question', '')}\n助手：{row.get('answer', '')}"
        for row in rows
    ]
    return "\n".join(messages)[-SESSION_CONTEXT_MAX_CHARS:]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bounded_error(value: str) -> str:
    return str(value or "")[:300]
