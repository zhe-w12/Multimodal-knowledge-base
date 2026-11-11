"""FastAPI service layer for the education RAG pipeline."""

from __future__ import annotations

import base64
import binascii
import asyncio
import hashlib
import json
import mimetypes
import os
import re
import zlib
from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, RLock
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Literal, Optional
from urllib.parse import quote
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from manufacturing_rag.auth.routes import (
    _auth_store,
    _require_browser_request,
    _require_csrf,
    router as auth_router,
)
from manufacturing_rag.auth.conversation_routes import router as conversation_router
from manufacturing_rag.auth.conversation_store import ConversationLocked, ConversationNotFound, ConversationStore
from manufacturing_rag.auth.store import AuthServiceUnavailable, Principal
from manufacturing_rag.auth.workspace_store import WorkspaceForbidden, WorkspaceNotFound, WorkspaceStore
from manufacturing_rag.auth.workspace_routes import admin_router as workspace_admin_router
from manufacturing_rag.auth.workspace_routes import router as workspace_router
from manufacturing_rag.adapters.ocr import BaiduOCRAdapter
from manufacturing_rag.adapters.llm import build_llm_router
from manufacturing_rag.adapters.embedding import HashEmbeddingAdapter, build_embedding_adapter
from manufacturing_rag.adapters.query_optimizer import AdaptiveQueryOptimizer
from manufacturing_rag.adapters.retriever import HybridRetriever, MilvusHybridRetriever
from manufacturing_rag.adapters.vector_store import MilvusVectorStore
from manufacturing_rag.chunking import ETLValidationReport, prepare_chunks
from manufacturing_rag.config import DEFAULT_CONFIG, RAGConfig
from manufacturing_rag.debug_log import chunks_for_debug, retrieval_results_for_debug, write_debug_log
from manufacturing_rag.etl import MultimodalETLPipeline
from manufacturing_rag.infrastructure import RuntimeInfrastructure
from manufacturing_rag.pipeline import ManufacturingRAGPipeline, contextualize_question, retrieve_with_query_variants
from manufacturing_rag.pdf_quality import (
    chunks_have_unusable_pdf_text,
    looks_like_spaced_pdf_garble,
    pdf_chunks_are_searchable,
)
from manufacturing_rag.schemas import DEFAULT_WORKSPACE_ID, DocumentChunk, RAGResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_PATH = PROJECT_ROOT / "examples" / "knowledge_base.jsonl"
FRONTEND_SOURCE_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_SOURCE_DIR / "dist"
FRONTEND_DIR = FRONTEND_DIST_DIR
MAX_PDF_UPLOAD_BYTES = 10 * 1024 * 1024
DOCUMENT_FILES_DIR = PROJECT_ROOT / "tmp" / "document_files"


def safe_document_filename(source: str) -> str:
    filename = str(source or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    return filename or "document"


def stored_document_path(workspace_id: str, source: str) -> Path:
    workspace_key = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:24]
    document_key = hashlib.sha256(f"{workspace_id}\\0{source}".encode("utf-8")).hexdigest()
    return DOCUMENT_FILES_DIR / workspace_key / document_key


def stored_document_metadata_path(workspace_id: str, source: str) -> Path:
    path = stored_document_path(workspace_id, source)
    return path.with_name(f"{path.name}.json")


def stored_document_filename(workspace_id: str, source: str) -> str:
    metadata_path = stored_document_metadata_path(workspace_id, source)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        filename = str(metadata.get("filename") or "")
        if filename:
            return safe_document_filename(filename)
    except (OSError, ValueError, TypeError):
        pass
    return safe_document_filename(source)


def save_stored_document(workspace_id: str, source: str, filename: str, content: bytes) -> None:
    path = stored_document_path(workspace_id, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    metadata_path = stored_document_metadata_path(workspace_id, source)
    temporary_metadata_path = metadata_path.with_name(f"{metadata_path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
        temporary_metadata_path.write_text(
            json.dumps({"source": source, "filename": safe_document_filename(filename)}, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_metadata_path.replace(metadata_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)


def remove_stored_document(workspace_id: str, source: str) -> None:
    for path in (
        stored_document_path(workspace_id, source),
        stored_document_metadata_path(workspace_id, source),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Knowledge-base deletion should still complete if an old file is already gone.
            pass


def workspace_chunk_id(workspace_id: str, chunk_id: str) -> str:
    """Keep IDs stable in the default workspace and unique in other workspaces."""

    if workspace_id == DEFAULT_WORKSPACE_ID:
        return chunk_id
    workspace_hash = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:16]
    chunk_hash = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()
    return f"ws-{workspace_hash}-{chunk_hash}"


def workspace_session_id(workspace_id: str, session_id: Optional[str]) -> Optional[str]:
    if session_id is None or workspace_id == DEFAULT_WORKSPACE_ID:
        return session_id
    session_hash = hashlib.sha256(f"{workspace_id}\0{session_id}".encode("utf-8")).hexdigest()
    return f"ws-{session_hash}"


def authenticated_session_id(user_id: str, session_id: Optional[str]) -> Optional[str]:
    if session_id is None:
        return None
    digest = hashlib.sha256(f"{user_id}\0{session_id}".encode("utf-8")).hexdigest()
    return f"user-{digest}"


def tag_chunks_for_workspace(chunks: Iterable[DocumentChunk], workspace_id: str) -> List[DocumentChunk]:
    rows = list(chunks)
    id_map = {chunk.id: workspace_chunk_id(workspace_id, chunk.id) for chunk in rows}
    for chunk in rows:
        chunk.id = id_map[chunk.id]
        if chunk.parent_id:
            chunk.parent_id = id_map.get(chunk.parent_id, workspace_chunk_id(workspace_id, chunk.parent_id))
        metadata = dict(chunk.metadata or {})
        metadata["workspace_id"] = workspace_id
        if metadata.get("parent_id"):
            parent_id = str(metadata["parent_id"])
            metadata["parent_id"] = id_map.get(parent_id, workspace_chunk_id(workspace_id, parent_id))
        chunk.metadata = metadata
    return rows


def sanitize_error_detail(value: Any) -> str:
    """Return a bounded diagnostic string without common credential values."""

    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text = re.sub(
        r"(?i)\b(token|api[_-]?key|secret|password|passwd)\b\s*[:=]\s*[^\s,;，。]+",
        lambda match: f"{match.group(1)}=<已隐藏>",
        text,
    )
    return text.strip()[:1200]


def explain_api_error(detail: Any, status_code: int, error_type: str = "") -> Dict[str, str]:
    """Translate operational exceptions into a stable user-facing reason and action."""

    technical_detail = sanitize_error_detail(detail)
    lowered = technical_detail.lower()
    reason = technical_detail or "服务未返回具体异常信息。"
    suggestion = "请稍后重试；若问题持续，请查看后端日志定位具体阶段。"
    code = f"HTTP_{status_code}"

    if any(term in lowered for term in ("fail connecting", "server unavailable", "illegal connection params", "milvusexception")):
        code = "VECTOR_CONNECTION_FAILED"
        reason = "无法连接本地 Docker Milvus（项目仅使用本地实例，不连接线上 Zilliz），可能是容器未就绪或 Endpoint 与 Token 不匹配。"
        suggestion = "请确认 Docker Milvus 容器健康运行，并检查 MILVUS_ENDPOINT、MILVUS_TOKEN 后重试。"
    elif "milvus configuration missing" in lowered or "zilliz configuration missing" in lowered or ("milvus" in lowered and "not configured" in lowered):
        code = "VECTOR_CONFIG_MISSING"
        reason = "本地 Docker Milvus 连接配置缺失，后端没有读取到 Endpoint 或 Token。"
        suggestion = "请在 .env 中配置 MILVUS_ENDPOINT 和 MILVUS_TOKEN，然后重启后端。"
    elif any(term in lowered for term in ("unauthenticated", "unauthorized", "authentication failed", "invalid token", "permission denied")):
        code = "AUTHENTICATION_FAILED"
        reason = "云服务身份验证失败，凭据无效、已过期或没有当前操作权限。"
        suggestion = "请重新复制对应服务的 Token/API Key，确认没有多余空格，并检查账号权限。"
    elif any(term in lowered for term in ("timed out", "timeout", "deadline exceeded")):
        code = "UPSTREAM_TIMEOUT"
        reason = "调用外部服务超时，网络较慢或云服务没有在规定时间内响应。"
        suggestion = "请检查网络后重试；处理大文件时可减少 PDF 页数或降低渲染 DPI。"
    elif any(term in lowered for term in ("name resolution", "getaddrinfo", "dns")):
        code = "DNS_FAILED"
        reason = "无法解析云服务域名，当前网络或 DNS 配置不可用。"
        suggestion = "请检查网络、DNS、代理和防火墙设置后重试。"
    elif "baidu ocr" in lowered or "百度 ocr" in lowered:
        code = "OCR_FAILED"
        reason = technical_detail
        suggestion = "请检查百度 OCR Key、图片/PDF 清晰度和网络；也可以粘贴手动 OCR 文本作为兜底。"
    elif status_code == 413 or any(term in lowered for term in ("exceeds 10mb", "too large", "upload limit")):
        code = "FILE_TOO_LARGE"
        reason = "上传文件超过系统允许的 10MB 大小限制。"
        suggestion = "请压缩或拆分文件后重新上传。"
    elif "base64" in lowered:
        code = "INVALID_FILE_PAYLOAD"
        reason = "上传文件的数据编码无效或传输内容不完整。"
        suggestion = "请重新选择原始文件上传，不要上传损坏或未完整下载的文件。"
    elif "not a pdf" in lowered or "不是 pdf" in lowered:
        code = "INVALID_PDF"
        reason = "上传内容不是有效的 PDF 文件。"
        suggestion = "请确认文件扩展名与实际格式一致，并重新导出 PDF 后上传。"
    elif "invalid json" in lowered or "jsondecode" in lowered:
        code = "INVALID_JSON"
        reason = f"JSON/JSONL 格式不正确：{technical_detail}"
        suggestion = "请检查报错行附近是否缺少引号、逗号或大括号，并确保 JSONL 每行是一个完整对象。"
    elif status_code == 422 or error_type == "RequestValidationError":
        code = "INVALID_REQUEST"
        reason = f"请求参数不符合接口要求：{technical_detail}"
        suggestion = "请检查必填字段、字段类型以及输入内容是否为空。"
    elif status_code == 404:
        code = "API_NOT_FOUND"
        reason = "请求的后端接口不存在，通常是 API 地址填写错误或前后端版本不一致。"
        suggestion = "请检查 API Endpoint，并确认后端服务已启动为当前项目版本。"
    elif status_code == 429:
        code = "RATE_LIMITED"
        reason = "云服务请求过于频繁或额度已用尽。"
        suggestion = "请稍后重试，并检查对应云服务的调用额度和限流设置。"
    elif status_code >= 500:
        code = "SERVER_ERROR"
        reason = technical_detail or "后端或上游云服务发生内部错误。"
        suggestion = "请稍后重试，并查看 run_api.err.log 与 logs/rag_debug.jsonl。"

    return {
        "code": code,
        "reason": reason,
        "suggestion": suggestion,
        "technical_detail": technical_detail,
    }


class RerankRuleRequest(BaseModel):
    boost_terms: List[str] = Field(default_factory=list, description="Terms that raise matching passages in the rerank order")
    exclude_terms: List[str] = Field(default_factory=list, description="Terms that remove matching passages from rerank results")


class AnswerRequest(BaseModel):
    """Question payload accepted by the frontend API mode."""

    question: str = Field(..., min_length=1, description="User question to answer from indexed OCR/CLIP materials")
    session_id: Optional[str] = Field(None, max_length=128, description="Conversation id cached in Redis and persisted in MySQL")
    conversation_id: Optional[str] = Field(None, max_length=36, description="Server-owned private conversation id")
    conversation_context: str = Field("", max_length=4000, description="Recent turns from the current chat")
    workspace_id: str = Field(DEFAULT_WORKSPACE_ID, min_length=1, max_length=128, description="Knowledge workspace to search")
    scope: Literal["all", "common", "workspace"] = Field(
        "workspace",
        description="Requested search scope; the server resolves this to workspaces the signed-in user can read",
    )
    rerank_rules: RerankRuleRequest = Field(default_factory=RerankRuleRequest)
    hallucination_threshold: float = Field(0.75, ge=0.0, le=0.95, description="Minimum confidence required to accept a source-checked answer")


class KnowledgeBaseReplaceRequest(BaseModel):
    """JSON knowledge-base update payload."""

    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="OCR structured chunks")
    mode: str = Field("append", description="append or replace; append is the safe default")
    workspace_id: str = Field(DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)


class JsonlIngestRequest(BaseModel):
    """JSONL text ingestion payload for browser or integration clients."""

    content: str = Field(..., min_length=1, description="JSONL content; one OCR chunk per line or a JSON array")
    mode: str = Field("append", description="append or replace; append is the safe default")
    source_name: Optional[str] = Field(None, description="Optional logical source label used in error messages")
    workspace_id: str = Field(DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)


class PdfIngestRequest(BaseModel):
    """Base64 PDF ingestion payload sent by the zero-dependency frontend."""

    filename: str = Field(..., min_length=1, description="Original PDF file name")
    content_base64: str = Field(..., min_length=1, description="Base64 encoded PDF bytes")
    mode: str = Field("append", description="append or replace; append is the safe default")
    max_pages: Optional[int] = Field(None, ge=0, le=1000, description="Maximum OCR pages; 0 means unlimited")
    render_dpi: Optional[int] = Field(None, ge=72, le=300, description="PDF page render DPI for OCR fallback")
    workspace_id: str = Field(DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)


class ImageOcrIngestRequest(BaseModel):
    """Base64 image ingestion payload for OCR processing."""

    filename: str = Field(..., min_length=1, description="Original image file name")
    content_base64: str = Field(..., min_length=1, description="Base64 encoded image bytes")
    ocr_text: Optional[str] = Field(None, description="Optional OCR text already extracted by client or another tool")
    mode: str = Field("append", description="append or replace; append is the safe default")
    workspace_id: str = Field(DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)


class DocumentIngestRequest(BaseModel):
    """Base64 PDF, Word, PowerPoint, text or image payload for the multimodal ETL chain."""

    filename: str = Field(..., min_length=1, description="Original file name with extension")
    content_base64: str = Field(..., min_length=1, description="Base64 encoded document bytes")
    mode: str = Field("append", description="append or replace; append is the safe default")
    max_pages: Optional[int] = Field(None, ge=0, le=1000, description="Maximum PDF OCR pages; 0 means unlimited")
    render_dpi: Optional[int] = Field(None, ge=72, le=300, description="PDF render DPI used by PaddleOCR fallback")
    workspace_id: str = Field(DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)


class DocumentDeleteRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Exact document source name returned by the knowledge-base summary")
    workspace_id: str = Field(DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)


class VectorCollectionRequest(BaseModel):
    """Zilliz collection creation request."""

    drop_existing: bool = Field(False, description="Drop and recreate the Zilliz collection")


class VectorIngestRequest(BaseModel):
    """Vector synchronization mode sent after a knowledge-base upload."""

    mode: str = Field("append", description="append or replace; append preserves existing remote rows")
    workspace_id: Optional[str] = Field(None, min_length=1, max_length=128)


class VectorSearchRequest(BaseModel):
    """Zilliz vector-search request."""

    query: str = Field(..., min_length=1, description="Vector search query")
    limit: int = Field(5, ge=1, le=50, description="Maximum result count")
    workspace_id: str = Field(DEFAULT_WORKSPACE_ID, min_length=1, max_length=128)


class HealthResponse(BaseModel):
    status: str
    service: str
    kb_chunk_count: int
    model_mode: str
    embedding_provider: str
    embedding_model: str
    rerank_provider: str
    rerank_model: str
    ocr_provider: str
    vector_store_provider: str
    vector_collection_name: str
    strict_source_required: bool
    default_kb_path: str
    query_classifier: str
    query_agent: str
    redis_status: str
    mysql_status: str


class KnowledgeBaseSummary(BaseModel):
    chunk_count: int
    sources: List[str]
    doc_types: List[str]
    review_chunk_count: int = 0
    parent_count: int = 0


class KnowledgeBaseUpdateResponse(BaseModel):
    status: str
    mode: str
    chunk_count: int
    summary: KnowledgeBaseSummary
    validation: Dict[str, int] = Field(default_factory=dict)
    parser_chain: List[str] = Field(default_factory=list)


class KnowledgeBaseState:
    """Thread-safe in-memory knowledge base and pipeline holder."""

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG, initial_chunks: Optional[Iterable[DocumentChunk]] = None) -> None:
        self._config = config
        self._lock = RLock()
        self._version = 1
        self.infrastructure = RuntimeInfrastructure(config)
        self._last_validation = ETLValidationReport(0, 0, 0)
        self._persistence_enabled = initial_chunks is None and bool(config.kb_runtime_path.strip())
        provided_chunks = list(initial_chunks) if initial_chunks is not None else []
        if provided_chunks:
            self._chunks = provided_chunks
        elif self._persistence_enabled:
            runtime_path = self._runtime_path()
            if runtime_path is not None and runtime_path.exists():
                # An existing empty runtime file means the user intentionally
                # cleared the knowledge base; do not repopulate bundled samples.
                self._chunks = self._load_runtime_chunks()
            else:
                self._chunks = self._load_startup_chunks()
        else:
            self._chunks = self._load_startup_chunks()
        self._workspace_pipelines: Dict[str, tuple[int, ManufacturingRAGPipeline]] = {}

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        rerank_rules: Optional[Dict[str, List[str]]] = None,
        conversation_context: str = "",
        on_token: Optional[Callable[[str], None]] = None,
        hallucination_threshold: float = 0.75,
        on_status: Optional[Callable[[str], None]] = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        workspace_ids: Optional[Iterable[str]] = None,
        principal_user_id: Optional[str] = None,
        permission_cache_scope: Optional[str] = None,
        strict_workspace_metadata: bool = False,
    ) -> RAGResponse:
        with self._lock:
            if workspace_ids is None:
                pipeline = self._workspace_pipeline_locked(workspace_id)
                effective_workspace_ids = [workspace_id]
            else:
                effective_workspace_ids = sorted(set(str(value) for value in workspace_ids if value))
                pipeline = self._workspace_scope_pipeline_locked(
                    effective_workspace_ids,
                    strict_workspace_metadata=strict_workspace_metadata,
                )
            knowledge_version = self._version
        scope_key = ",".join(effective_workspace_ids)
        scoped_session_id = workspace_session_id(scope_key, session_id)
        stored_context = self.infrastructure.get_session_context(scoped_session_id)
        session_context = str(conversation_context or "").strip()[-4000:] or stored_context
        if on_status:
            on_status("正在识别问题类型…")
        intent = pipeline.classify_intent(question, session_context)
        cache_basis = contextualize_question(question, session_context)
        safe_rerank_rules = rerank_rules or {"boost_terms": [], "exclude_terms": []}
        safe_threshold = min(0.95, max(0.0, float(hallucination_threshold)))
        cache_basis = (
            f"{cache_basis}\n[intent_category:{intent.category}]"
            f"\n[workspace_ids:{','.join(effective_workspace_ids)}]"
            f"\n[principal_user_id:{principal_user_id or 'anonymous'}]"
            f"\n[permission_cache_scope:{permission_cache_scope or 'legacy'}]"
            f"\n[rerank_rules:{json.dumps(safe_rerank_rules, ensure_ascii=False, sort_keys=True)}]"
            f"\n[hallucination_threshold:{safe_threshold:.2f}]"
        )
        # Chat replies depend on conversation tone and should not be replaced
        # by a previous global cache entry. They are still persisted into the
        # selected session's conversation history below.
        cached = None
        if intent.allowed and intent.category != "chit_chat":
            cached = self.infrastructure.get_cached_answer(cache_basis, knowledge_version)
        if cached is not None and cached.hallucination_check and cached.hallucination_check.passed:
            self.infrastructure.cache_answer(cached, knowledge_version, session_id=scoped_session_id, cache_basis=cache_basis)
            return cached
        response = pipeline.answer(
            question,
            conversation_context=session_context,
            rerank_rules=safe_rerank_rules,
            on_token=on_token,
            hallucination_threshold=safe_threshold,
            on_status=on_status,
            classified_intent=intent,
        )
        self.infrastructure.cache_answer(response, knowledge_version, session_id=scoped_session_id, cache_basis=cache_basis)
        return response

    def _chunk_workspace_id(self, chunk: DocumentChunk) -> str:
        return str((chunk.metadata or {}).get("workspace_id") or DEFAULT_WORKSPACE_ID)

    def _workspace_pipeline_locked(self, workspace_id: str) -> ManufacturingRAGPipeline:
        cached = self._workspace_pipelines.get(workspace_id)
        if cached and cached[0] == self._version:
            return cached[1]
        chunks = [chunk for chunk in self._chunks if self._chunk_workspace_id(chunk) == workspace_id]
        pipeline = ManufacturingRAGPipeline(
            chunks,
            config=self._config,
            remote_cache_callback=self._cache_remote_chunks,
            workspace_id=workspace_id,
        )
        self._workspace_pipelines[workspace_id] = (self._version, pipeline)
        return pipeline

    def _workspace_scope_pipeline_locked(
        self,
        workspace_ids: Iterable[str],
        *,
        strict_workspace_metadata: bool = False,
    ) -> ManufacturingRAGPipeline:
        normalized_ids = tuple(sorted(set(str(value) for value in workspace_ids if value)))
        if not normalized_ids:
            raise HTTPException(status_code=403, detail="没有可检索的工作区")
        if len(normalized_ids) == 1 and not strict_workspace_metadata:
            return self._workspace_pipeline_locked(normalized_ids[0])
        cache_key = "scope:" + hashlib.sha256("\0".join(normalized_ids).encode("utf-8")).hexdigest()
        cached = self._workspace_pipelines.get(cache_key)
        if cached and cached[0] == self._version:
            return cached[1]
        allowed = frozenset(normalized_ids)
        chunks = [chunk for chunk in self._chunks if self._chunk_workspace_id(chunk) in allowed]
        pipeline = ManufacturingRAGPipeline(
            chunks,
            config=self._config,
            remote_cache_callback=self._cache_remote_chunks,
            workspace_ids=normalized_ids,
            strict_workspace_metadata=strict_workspace_metadata,
        )
        self._workspace_pipelines[cache_key] = (self._version, pipeline)
        return pipeline

    def chunks_snapshot(self, workspace_id: Optional[str] = None) -> List[DocumentChunk]:
        with self._lock:
            if workspace_id is None:
                return list(self._chunks)
            return [chunk for chunk in self._chunks if self._chunk_workspace_id(chunk) == workspace_id]

    def document_chunk_ids(self, source: str, workspace_id: str = DEFAULT_WORKSPACE_ID) -> List[str]:
        with self._lock:
            return [
                chunk.id for chunk in self._chunks
                if chunk.source == source and self._chunk_workspace_id(chunk) == workspace_id
            ]

    def list_original_documents(self, workspace_id: str = DEFAULT_WORKSPACE_ID) -> Dict[str, Any]:
        workspace_key = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:24]
        workspace_dir = DOCUMENT_FILES_DIR / workspace_key
        documents = []
        if not workspace_dir.is_dir():
            return {"documents": documents, "count": 0, "workspace_id": workspace_id}
        for metadata_path in workspace_dir.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                source = str(metadata.get("source") or "")
                filename = safe_document_filename(str(metadata.get("filename") or source))
                if not source or metadata_path != stored_document_metadata_path(workspace_id, source):
                    continue
                path = stored_document_path(workspace_id, source)
                if not path.is_file():
                    continue
                documents.append({
                    "source": source,
                    "filename": filename,
                    "size_bytes": path.stat().st_size,
                })
            except (OSError, ValueError, TypeError):
                continue
        documents.sort(key=lambda item: item["filename"].casefold())
        return {"documents": documents, "count": len(documents), "workspace_id": workspace_id}

    def document_content(
        self,
        source: str,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        max_chars: int = 150_000,
    ) -> Dict[str, Any]:
        with self._lock:
            chunks = [
                chunk for chunk in self._chunks
                if chunk.source == source and self._chunk_workspace_id(chunk) == workspace_id
            ]
            if not chunks:
                raise HTTPException(status_code=404, detail=f"知识库中不存在资料：{source}")
            content = "\n\n".join(chunk.text.strip() for chunk in chunks if chunk.text and chunk.text.strip())
            safe_limit = max(0, min(int(max_chars), 500_000))
            truncated = bool(safe_limit and len(content) > safe_limit)
            if truncated:
                content = content[:safe_limit]
            return {
                "source": source,
                "filename": stored_document_filename(workspace_id, source),
                "content": content,
                "chunk_count": len(chunks),
                "original_available": stored_document_path(workspace_id, source).is_file(),
                "truncated": truncated,
            }

    def workspace_chunk_ids(self, workspace_id: str) -> List[str]:
        with self._lock:
            return [
                chunk.id for chunk in self._chunks
                if self._chunk_workspace_id(chunk) == workspace_id
            ]

    def delete_workspace(self, workspace_id: str) -> Dict[str, Any]:
        if workspace_id == DEFAULT_WORKSPACE_ID:
            raise HTTPException(status_code=400, detail="默认工作区不能删除")

        workspace_key = hashlib.sha256(workspace_id.encode("utf-8")).hexdigest()[:24]
        workspace_dir = DOCUMENT_FILES_DIR / workspace_key
        with self._lock:
            removed = [chunk for chunk in self._chunks if self._chunk_workspace_id(chunk) == workspace_id]
            removed_chunk_count = len(removed)
            removed_document_count = 0

            try:
                if workspace_dir.is_symlink():
                    workspace_dir.unlink(missing_ok=True)
                elif workspace_dir.is_dir():
                    stored_files = list(workspace_dir.iterdir())
                    removed_document_count = sum(1 for path in stored_files if path.suffix == ".json")
                    for path in stored_files:
                        if path.is_file() or path.is_symlink():
                            path.unlink(missing_ok=True)
                        else:
                            raise OSError("unexpected directory in workspace document store")
                    workspace_dir.rmdir()
            except OSError as exc:
                raise HTTPException(status_code=500, detail="无法清理工作区中的原始文档") from exc

            next_chunks = [
                chunk for chunk in self._chunks
                if self._chunk_workspace_id(chunk) != workspace_id
            ]
            self._persist_runtime_chunks_locked(next_chunks, strict=True)
            self._chunks = next_chunks
            self._version += 1
            self._invalidate_workspace_pipelines_locked()
            write_debug_log(
                self._config,
                "kb_workspace_deleted",
                workspace_id=workspace_id,
                deleted_chunk_count=removed_chunk_count,
                deleted_document_count=removed_document_count,
            )
            return {
                "workspace_id": workspace_id,
                "removed_chunk_count": removed_chunk_count,
                "removed_document_count": removed_document_count,
            }

    def delete_document(
        self,
        source: str,
        expected_chunk_ids: Iterable[str],
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> Dict[str, Any]:
        expected_ids = set(expected_chunk_ids)
        with self._lock:
            removed = [
                chunk for chunk in self._chunks
                if chunk.source == source and chunk.id in expected_ids and self._chunk_workspace_id(chunk) == workspace_id
            ]
            if not removed:
                raise HTTPException(status_code=404, detail=f"知识库中不存在资料：{source}")
            removed_ids = {chunk.id for chunk in removed}
            self._chunks = [
                chunk for chunk in self._chunks
                if not (
                    chunk.source == source
                    and chunk.id in removed_ids
                    and self._chunk_workspace_id(chunk) == workspace_id
                )
            ]
            remove_stored_document(workspace_id, source)
            self._version += 1
            self._persist_runtime_chunks_locked()
            self._invalidate_workspace_pipelines_locked()
            summary = self.summary_locked(workspace_id)
            write_debug_log(
                self._config,
                "kb_document_deleted",
                source=source,
                deleted_chunk_count=len(removed),
                remaining_chunk_count=sum(1 for chunk in self._chunks if self._chunk_workspace_id(chunk) == workspace_id),
            )
            return {
                "source": source,
                "removed_chunk_count": len(removed),
                "summary": summary.model_dump() if hasattr(summary, "model_dump") else summary.dict(),
            }

    def list_chunks(
        self,
        limit: int = 50,
        offset: int = 0,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            safe_offset = max(offset, 0)
            safe_limit = max(min(limit, 200), 1)
            chunks = self._chunks if workspace_id is None else [
                chunk for chunk in self._chunks if self._chunk_workspace_id(chunk) == workspace_id
            ]
            rows = chunks[safe_offset : safe_offset + safe_limit]
            return {
                "total": len(chunks),
                "offset": safe_offset,
                "limit": safe_limit,
                "chunks": [chunk_to_dict(chunk) for chunk in rows],
            }

    def update_chunks(
        self,
        payloads: Iterable[Dict[str, Any]],
        mode: str = "append",
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> KnowledgeBaseUpdateResponse:
        normalized_mode = validate_mode(mode)
        chunks, report = prepare_chunks((DocumentChunk.from_mapping(payload) for payload in payloads), self._config)
        log_chunks_for_ingest(self._config, "kb_chunks_payload", chunks, normalized_mode)
        return self._apply_chunks_locked(chunks, normalized_mode, validation=report, workspace_id=workspace_id)

    def ingest_pdf(
        self,
        filename: str,
        content_base64: str,
        mode: str = "append",
        max_pages: Optional[int] = None,
        render_dpi: Optional[int] = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> KnowledgeBaseUpdateResponse:
        normalized_mode = validate_mode(mode)
        pdf_bytes = decode_pdf_base64(content_base64)
        raw_chunks = pdf_bytes_to_chunks(
            pdf_bytes,
            filename=filename,
            config=self._config,
            max_pages=max_pages,
            render_dpi=render_dpi,
        )
        chunks, report = prepare_chunks(raw_chunks, self._config)
        log_chunks_for_ingest(self._config, "pdf_ingest_payload", chunks, normalized_mode, filename=filename)
        return self._apply_chunks_locked(
            chunks,
            normalized_mode,
            validation=report,
            parser_chain=["PDF文本解析", self._config.ocr_provider],
            workspace_id=workspace_id,
        )

    def ingest_image_ocr(
        self,
        filename: str,
        content_base64: str,
        ocr_text: Optional[str] = None,
        mode: str = "append",
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> KnowledgeBaseUpdateResponse:
        normalized_mode = validate_mode(mode)
        raw_chunks = image_payload_to_chunks(
            filename=filename,
            content_base64=content_base64,
            ocr_text=ocr_text,
            config=self._config,
        )
        chunks, report = prepare_chunks(raw_chunks, self._config)
        log_chunks_for_ingest(self._config, "image_ocr_ingest_payload", chunks, normalized_mode, filename=filename)
        return self._apply_chunks_locked(
            chunks,
            normalized_mode,
            validation=report,
            parser_chain=[self._config.ocr_provider, "Qwen-VL"],
            workspace_id=workspace_id,
        )

    def ingest_document(
        self,
        filename: str,
        content_base64: str,
        mode: str = "append",
        max_pages: Optional[int] = None,
        render_dpi: Optional[int] = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> KnowledgeBaseUpdateResponse:
        normalized_mode = validate_mode(mode)
        content = decode_document_base64(content_base64)
        etl_config = replace(
            self._config,
            baidu_ocr_pdf_max_pages=self._config.baidu_ocr_pdf_max_pages if max_pages is None else max_pages,
            baidu_ocr_pdf_render_dpi=self._config.baidu_ocr_pdf_render_dpi if render_dpi is None else render_dpi,
        )
        try:
            result = MultimodalETLPipeline(etl_config).parse_bytes(
                content,
                filename=filename,
                max_pages=max_pages,
                render_dpi=render_dpi,
            )
        except Exception as exc:  # noqa: BLE001 - convert parser-specific failures to a stable API error.
            raise HTTPException(status_code=400, detail=f"多模态 ETL 失败：{exc}") from exc
        log_chunks_for_ingest(self._config, "multimodal_document_ingest", result.chunks, normalized_mode, filename=filename)
        response = self._apply_chunks_locked(
            result.chunks,
            normalized_mode,
            validation=result.validation,
            parser_chain=result.parser_chain,
            workspace_id=workspace_id,
        )
        document_source = next((chunk.source for chunk in result.chunks if chunk.source), filename)
        save_stored_document(workspace_id, document_source, filename, content)
        return response

    def _apply_chunks_locked(
        self,
        chunks: List[DocumentChunk],
        normalized_mode: str,
        validation: Optional[ETLValidationReport] = None,
        parser_chain: Optional[List[str]] = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> KnowledgeBaseUpdateResponse:
        if not chunks and normalized_mode == "replace":
            raise HTTPException(status_code=400, detail="replace mode requires at least one chunk")
        chunks = tag_chunks_for_workspace(chunks, workspace_id)
        with self._lock:
            if normalized_mode == "replace":
                previous_sources = {
                    chunk.source for chunk in self._chunks
                    if self._chunk_workspace_id(chunk) == workspace_id and chunk.source
                }
                incoming_sources = {chunk.source for chunk in chunks if chunk.source}
                self._chunks = [
                    chunk for chunk in self._chunks if self._chunk_workspace_id(chunk) != workspace_id
                ] + chunks
                for removed_source in previous_sources - incoming_sources:
                    remove_stored_document(workspace_id, removed_source)
            else:
                # Keep different documents searchable while treating a
                # repeated upload of the same source as an update. This
                # avoids stale pages and duplicate chunks in append mode.
                incoming_by_id = {chunk.id: chunk for chunk in chunks}
                incoming_chunks = list(incoming_by_id.values())
                incoming_sources = {
                    chunk.source
                    for chunk in incoming_chunks
                    if chunk.source and chunk.source != "unknown"
                }
                self._chunks = [
                    chunk
                    for chunk in self._chunks
                    if self._chunk_workspace_id(chunk) != workspace_id
                    or (chunk.id not in incoming_by_id and chunk.source not in incoming_sources)
                ]
                self._chunks.extend(incoming_chunks)
            self._version += 1
            self._last_validation = validation or ETLValidationReport(len(chunks), 0, 0)
            self._persist_runtime_chunks_locked()
            log_chunks_for_ingest(self._config, "kb_before_pipeline_rebuild", self._chunks, normalized_mode)
            self._invalidate_workspace_pipelines_locked()
            return KnowledgeBaseUpdateResponse(
                status="ok",
                mode=normalized_mode,
                chunk_count=sum(1 for chunk in self._chunks if self._chunk_workspace_id(chunk) == workspace_id),
                summary=self.summary_locked(workspace_id),
                validation=self._last_validation.to_dict(),
                parser_chain=list(parser_chain or []),
            )

    def ingest_jsonl(
        self,
        content: str,
        mode: str = "append",
        source_name: Optional[str] = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> KnowledgeBaseUpdateResponse:
        payloads = parse_jsonl_or_array(content, source_name=source_name or "request.content")
        return self.update_chunks(payloads, mode=mode, workspace_id=workspace_id)

    def summary(self, workspace_id: Optional[str] = DEFAULT_WORKSPACE_ID) -> KnowledgeBaseSummary:
        with self._lock:
            return self.summary_locked(workspace_id)

    def summary_locked(self, workspace_id: Optional[str] = DEFAULT_WORKSPACE_ID) -> KnowledgeBaseSummary:
        chunks = self._chunks if workspace_id is None else [
            chunk for chunk in self._chunks if self._chunk_workspace_id(chunk) == workspace_id
        ]
        sources = sorted({chunk.source for chunk in chunks if chunk.source})
        doc_types = sorted({str(chunk.metadata.get("doc_type")) for chunk in chunks if chunk.metadata.get("doc_type")})
        parent_ids = {chunk.parent_id for chunk in chunks if chunk.parent_id}
        return KnowledgeBaseSummary(
            chunk_count=len(chunks),
            sources=sources,
            doc_types=doc_types,
            review_chunk_count=sum(1 for chunk in chunks if chunk.needs_review),
            parent_count=len(parent_ids),
        )

    def _invalidate_workspace_pipelines_locked(self) -> None:
        self._workspace_pipelines.clear()

    def _cache_remote_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        """Persist validated remote contexts for later local-first queries."""

        incoming_by_id = {chunk.id: chunk for chunk in chunks if chunk.id}
        if not incoming_by_id:
            return
        with self._lock:
            current_by_id = {chunk.id: chunk for chunk in self._chunks}
            changed = [
                chunk
                for chunk_id, chunk in incoming_by_id.items()
                if current_by_id.get(chunk_id) != chunk
            ]
            if not changed:
                return
            current_by_id.update(incoming_by_id)
            self._chunks = list(current_by_id.values())
            self._version += 1
            self._invalidate_workspace_pipelines_locked()
            self._persist_runtime_chunks_locked()
            write_debug_log(
                self._config,
                "kb_remote_context_cache",
                cached_count=len(changed),
                total_count=len(self._chunks),
                chunks=chunks_for_debug(changed, text_chars=self._config.debug_log_text_chars),
            )

    def _load_startup_chunks(self) -> List[DocumentChunk]:
        paths = startup_kb_paths()
        chunks: List[DocumentChunk] = []
        for path in paths:
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8-sig")
            chunks.extend(DocumentChunk.from_mapping(payload) for payload in parse_jsonl_or_array(content, source_name=str(path)))
        return chunks

    def _runtime_path(self) -> Optional[Path]:
        raw_path = self._config.kb_runtime_path.strip()
        if not raw_path:
            return None
        return Path(raw_path).expanduser().resolve()

    def _load_runtime_chunks(self) -> List[DocumentChunk]:
        path = self._runtime_path()
        if path is None or not path.exists():
            return []
        try:
            content = path.read_text(encoding="utf-8-sig")
            return [
                DocumentChunk.from_mapping(payload)
                for payload in parse_jsonl_or_array(content, source_name=str(path))
            ]
        except (OSError, ValueError, HTTPException) as exc:
            write_debug_log(self._config, "kb_runtime_load_error", path=str(path), error=str(exc))
            return []

    def _persist_runtime_chunks_locked(
        self,
        chunks: Optional[List[DocumentChunk]] = None,
        *,
        strict: bool = False,
    ) -> None:
        path = self._runtime_path()
        if not self._persistence_enabled or path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_name(f"{path.name}.tmp")
            rows = self._chunks if chunks is None else chunks
            payload = "\n".join(json.dumps(chunk_to_dict(chunk), ensure_ascii=False) for chunk in rows)
            temporary_path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")
            os.replace(temporary_path, path)
        except OSError as exc:
            write_debug_log(self._config, "kb_runtime_persist_error", path=str(path), error=str(exc))
            if strict:
                raise HTTPException(status_code=503, detail="无法更新工作区索引文件，请检查磁盘空间和权限后重试") from exc


def startup_kb_paths() -> List[Path]:
    configured = os.getenv("RAG_KB_PATHS", "").strip()
    if configured:
        return [Path(item).expanduser().resolve() for item in configured.split(";") if item.strip()]
    return [DEFAULT_KB_PATH]


def validate_mode(mode: str) -> str:
    normalized = (mode or "append").strip().lower()
    if normalized not in {"replace", "append"}:
        raise HTTPException(status_code=400, detail="mode must be 'replace' or 'append'")
    return normalized


def log_chunks_for_ingest(
    config: RAGConfig,
    event: str,
    chunks: List[DocumentChunk],
    mode: str,
    filename: Optional[str] = None,
) -> None:
    write_debug_log(
        config,
        event,
        mode=mode,
        filename=filename,
        chunk_count=len(chunks),
        chunks=chunks_for_debug(chunks, text_chars=config.debug_log_text_chars),
    )


def parse_jsonl_or_array(content: str, source_name: str = "content") -> List[Dict[str, Any]]:
    text = (content or "").strip().lstrip("\ufeff")
    if not text:
        raise HTTPException(status_code=400, detail="knowledge content is empty")
    try:
        if text.startswith("["):
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError("JSON array expected")
            return [ensure_mapping(item, source_name, index + 1) for index, item in enumerate(payload)]
        records: List[Dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip().lstrip("\ufeff")
            if not stripped:
                continue
            records.append(ensure_mapping(json.loads(stripped), source_name, line_number))
        if not records:
            raise ValueError("no JSONL records found")
        return records
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON in {source_name}:{exc.lineno}: {exc.msg}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid knowledge content in {source_name}: {exc}") from exc


def decode_pdf_base64(content_base64: str) -> bytes:
    payload = (content_base64 or "").strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        pdf_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid base64 PDF payload") from exc
    if len(pdf_bytes) > MAX_PDF_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 10MB upload limit")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="uploaded file is not a PDF")
    return pdf_bytes


def pdf_bytes_to_chunks(
    pdf_bytes: bytes,
    filename: str,
    config: RAGConfig = DEFAULT_CONFIG,
    max_pages: Optional[int] = None,
    render_dpi: Optional[int] = None,
) -> List[DocumentChunk]:
    source = Path(filename or "uploaded.pdf").name or "uploaded.pdf"
    page_texts = extract_pdf_text_by_page(pdf_bytes)
    chunks: List[DocumentChunk] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        for block_number, block_text in enumerate(split_text_blocks(page_text), start=1):
            chunks.append(
                DocumentChunk(
                    id=f"{source}:p{page_number}:b{block_number}",
                    text=block_text,
                    source=source,
                    page=page_number,
                    block_id=f"p{page_number}-b{block_number}",
                    modality="pdf_text",
                    metadata={"doc_type": "PDF文档", "ingest_format": "pdf", "parser": "lightweight_pdf_text"},
                )
            )
    is_searchable = bool(chunks) and pdf_chunks_are_searchable(chunks)
    write_debug_log(
        config,
        "pdf_lightweight_parse_result",
        filename=source,
        chunk_count=len(chunks),
        searchable=is_searchable,
        chunks=chunks_for_debug(chunks, text_chars=config.debug_log_text_chars),
    )
    if is_searchable:
        return chunks
    write_debug_log(config, "pdf_fallback_to_baidu_ocr", filename=source, reason="lightweight_parse_empty_or_garbled")
    chunks = pdf_bytes_to_baidu_ocr_chunks(
        pdf_bytes,
        source=source,
        config=config,
        max_pages=max_pages,
        render_dpi=render_dpi,
    )
    write_debug_log(
        config,
        "pdf_baidu_ocr_result",
        filename=source,
        chunk_count=len(chunks),
        ingest_formats=sorted({str(chunk.metadata.get("ingest_format")) for chunk in chunks if chunk.metadata.get("ingest_format")}),
        chunks=chunks_for_debug(chunks, text_chars=config.debug_log_text_chars),
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="百度 OCR 未从 PDF 中识别到可入库文本，请检查 PDF 清晰度。")
    return chunks


def pdf_bytes_to_baidu_ocr_chunks(
    pdf_bytes: bytes,
    source: str,
    config: RAGConfig = DEFAULT_CONFIG,
    max_pages: Optional[int] = None,
    render_dpi: Optional[int] = None,
) -> List[DocumentChunk]:
    try:
        page_limit = config.baidu_ocr_pdf_max_pages if max_pages is None else max_pages
        dpi = config.baidu_ocr_pdf_render_dpi if render_dpi is None else render_dpi
        return BaiduOCRAdapter(config).parse_pdf_bytes(
            pdf_bytes,
            source=source,
            max_pages=page_limit,
            render_dpi=dpi,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"PDF 文本解析质量不足，且百度 OCR 未完成：{exc}。请检查百度 OCR Key，或上传清晰图片/手动 OCR 文本。",
        ) from exc


def image_payload_to_chunks(
    filename: str,
    content_base64: str,
    ocr_text: Optional[str] = None,
    config: RAGConfig = DEFAULT_CONFIG,
) -> List[DocumentChunk]:
    source = Path(filename or "uploaded-image").name or "uploaded-image"
    image_bytes = decode_image_base64(content_base64)
    text = (ocr_text or "").strip()
    if text:
        return [
            DocumentChunk(
                id=f"{source}:image-ocr:1",
                text=text,
                source=source,
                page=1,
                block_id="image-ocr-1",
                modality="image_ocr",
                metadata={"doc_type": "图片OCR", "ingest_format": "image", "ocr_provider": "manual", "image_bytes": len(image_bytes)},
            )
        ]
    try:
        chunks = BaiduOCRAdapter(config).parse_image_bytes(image_bytes, source=source)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"百度 OCR 未完成：{exc}。请检查 OCR 凭据，或临时粘贴 OCR 文本。",
        ) from exc
    if not chunks:
        raise HTTPException(status_code=400, detail="百度 OCR 未识别到可入库文本，请检查图片清晰度。")
    return chunks


def decode_image_base64(content_base64: str) -> bytes:
    payload = (content_base64 or "").strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid base64 image payload") from exc
    if len(image_bytes) > MAX_PDF_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds 10MB upload limit")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="uploaded image is empty")
    return image_bytes


def decode_document_base64(content_base64: str) -> bytes:
    payload = (content_base64 or "").strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        content = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid base64 document payload") from exc
    if len(content) > MAX_PDF_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="document exceeds 10MB upload limit")
    if not content:
        raise HTTPException(status_code=400, detail="uploaded document is empty")
    return content


def extract_pdf_text_by_page(pdf_bytes: bytes) -> List[str]:
    objects = parse_pdf_objects(pdf_bytes)
    unicode_map = build_pdf_unicode_map(objects.values())
    page_refs = [obj_id for obj_id, body in objects.items() if is_pdf_page_object(body)]
    page_texts: List[str] = []
    for page_ref in page_refs:
        stream_refs = page_content_refs(objects[page_ref])
        page_parts = [extract_pdf_stream_text(objects.get(ref, b""), unicode_map) for ref in stream_refs]
        page_text = normalize_pdf_text("\n".join(part for part in page_parts if part))
        if page_text:
            page_texts.append(page_text)
    if page_texts:
        return page_texts
    stream_texts = [extract_pdf_stream_text(body, unicode_map) for body in objects.values()]
    merged = normalize_pdf_text("\n".join(text for text in stream_texts if text))
    return [merged] if merged else []


def parse_pdf_objects(pdf_bytes: bytes) -> Dict[int, bytes]:
    objects: Dict[int, bytes] = {}
    pattern = re.compile(rb"(\d+)\s+\d+\s+obj\b(.*?)\bendobj", re.S)
    for match in pattern.finditer(pdf_bytes):
        objects[int(match.group(1))] = match.group(2)
    return objects


def is_pdf_page_object(body: bytes) -> bool:
    return b"/Type" in body and re.search(rb"/Type\s*/Page\b", body) is not None


def page_content_refs(page_body: bytes) -> List[int]:
    contents_match = re.search(rb"/Contents\s*(\[(.*?)\]|\d+\s+\d+\s+R)", page_body, re.S)
    if not contents_match:
        return []
    content_spec = contents_match.group(1)
    return [int(ref) for ref in re.findall(rb"(\d+)\s+\d+\s+R", content_spec)]


def extract_pdf_stream_text(object_body: bytes, unicode_map: Dict[bytes, str]) -> str:
    stream = decoded_pdf_stream(object_body)
    if not stream:
        return ""
    parts: List[str] = []
    for text_block in re.findall(rb"BT\b(.*?)\bET", stream, flags=re.S):
        parts.extend(extract_pdf_strings(text_block, unicode_map))
    return normalize_pdf_text(" ".join(parts))


def decoded_pdf_stream(object_body: bytes) -> bytes:
    match = re.search(rb"stream\r?\n(.*?)\r?\nendstream", object_body, re.S)
    if not match:
        return b""
    stream = match.group(1)
    if b"/FlateDecode" in object_body:
        try:
            return zlib.decompress(stream)
        except zlib.error:
            return b""
    return stream


def build_pdf_unicode_map(object_bodies: Iterable[bytes]) -> Dict[bytes, str]:
    unicode_map: Dict[bytes, str] = {}
    for body in object_bodies:
        stream = decoded_pdf_stream(body)
        if b"beginbfchar" not in stream and b"beginbfrange" not in stream:
            continue
        text = stream.decode("latin-1", errors="ignore")
        for block in re.findall(r"beginbfchar(.*?)endbfchar", text, flags=re.S):
            for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>", block):
                unicode_map[bytes.fromhex(src)] = decode_utf16_hex(dst)
        for block in re.findall(r"beginbfrange(.*?)endbfrange", text, flags=re.S):
            for src_start, src_end, dst in re.findall(r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>", block):
                start = int(src_start, 16)
                end = int(src_end, 16)
                dst_start = int(dst, 16)
                src_width = len(src_start) // 2
                dst_width = len(dst) // 2
                if end - start > 512:
                    continue
                for offset, src_value in enumerate(range(start, end + 1)):
                    unicode_map[src_value.to_bytes(src_width, "big")] = decode_utf16_bytes(
                        (dst_start + offset).to_bytes(dst_width, "big")
                    )
            for src_start, src_end, dst_values in re.findall(
                r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>\s+\[(.*?)\]", block, flags=re.S
            ):
                start = int(src_start, 16)
                src_width = len(src_start) // 2
                values = re.findall(r"<([0-9A-Fa-f]+)>", dst_values)
                for offset, dst in enumerate(values[:512]):
                    unicode_map[(start + offset).to_bytes(src_width, "big")] = decode_utf16_hex(dst)
    return unicode_map


def extract_pdf_strings(text_block: bytes, unicode_map: Dict[bytes, str]) -> List[str]:
    strings: List[str] = []
    index = 0
    while index < len(text_block):
        char = text_block[index : index + 1]
        if char == b"(":
            literal, index = read_pdf_literal_string(text_block, index + 1)
            strings.append(decode_pdf_bytes(literal))
            continue
        if char == b"<" and text_block[index : index + 2] != b"<<":
            end = text_block.find(b">", index + 1)
            if end == -1:
                break
            hex_text = re.sub(rb"\s+", b"", text_block[index + 1 : end])
            try:
                raw = bytes.fromhex(hex_text.decode("ascii"))
                strings.append(decode_pdf_hex_bytes(raw, unicode_map))
            except ValueError:
                pass
            index = end + 1
            continue
        index += 1
    return [item for item in strings if item]


def read_pdf_literal_string(data: bytes, start: int) -> tuple[bytes, int]:
    result = bytearray()
    depth = 1
    index = start
    while index < len(data) and depth > 0:
        char = data[index]
        if char == 92:
            if index + 1 >= len(data):
                break
            next_char = data[index + 1]
            escape_map = {110: 10, 114: 13, 116: 9, 98: 8, 102: 12, 40: 40, 41: 41, 92: 92}
            if next_char in escape_map:
                result.append(escape_map[next_char])
                index += 2
                continue
            if 48 <= next_char <= 55:
                octal = bytes([next_char])
                lookahead = index + 2
                while lookahead < len(data) and len(octal) < 3 and 48 <= data[lookahead] <= 55:
                    octal += bytes([data[lookahead]])
                    lookahead += 1
                # PDF literal strings are byte-oriented. A few generators emit
                # malformed octal escapes above \377; keep the low byte so one
                # bad escape does not abort the complete document upload.
                result.append(int(octal, 8) & 0xFF)
                index = lookahead
                continue
            index += 2
            continue
        if char == 40:
            depth += 1
        elif char == 41:
            depth -= 1
            if depth == 0:
                index += 1
                break
        result.append(char)
        index += 1
    return bytes(result), index


def decode_pdf_hex_bytes(raw: bytes, unicode_map: Dict[bytes, str]) -> str:
    if unicode_map:
        lengths = sorted({len(key) for key in unicode_map}, reverse=True)
        output: List[str] = []
        index = 0
        while index < len(raw):
            matched = False
            for length in lengths:
                token = raw[index : index + length]
                if token in unicode_map:
                    output.append(unicode_map[token])
                    index += length
                    matched = True
                    break
            if not matched:
                output.append(decode_pdf_bytes(raw[index : index + 1]))
                index += 1
        return "".join(output)
    return decode_pdf_bytes(raw)


def decode_pdf_bytes(raw: bytes) -> str:
    if not raw:
        return ""
    for encoding in ("utf-8", "utf-16-be", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if text and printable_ratio(text) >= 0.5:
            return text
    return raw.decode("latin-1", errors="ignore")


def decode_utf16_hex(value: str) -> str:
    return decode_utf16_bytes(bytes.fromhex(value))


def decode_utf16_bytes(value: bytes) -> str:
    if not value:
        return ""
    encoding = "utf-16-be" if len(value) % 2 == 0 else "latin-1"
    return value.decode(encoding, errors="ignore")


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for char in text if char.isprintable() or char.isspace())
    return printable / len(text)


def normalize_pdf_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_blocks(text: str, max_chars: int = 900) -> List[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    blocks: List[str] = []
    current = ""
    for paragraph in paragraphs or [text.strip()]:
        if len(current) + len(paragraph) + 1 <= max_chars:
            current = f"{current}\n{paragraph}".strip()
            continue
        if current:
            blocks.append(current)
        while len(paragraph) > max_chars:
            blocks.append(paragraph[:max_chars].strip())
            paragraph = paragraph[max_chars:].strip()
        current = paragraph
    if current:
        blocks.append(current)
    return blocks


def ensure_mapping(value: Any, source_name: str, line_number: int) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"record {line_number} in {source_name} must be a JSON object")
    return value


def chunk_to_dict(chunk: DocumentChunk) -> Dict[str, Any]:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "page": chunk.page,
        "block_id": chunk.block_id,
        "modality": chunk.modality,
        "coordinates": None
        if chunk.coordinates is None
        else {
            "x1": chunk.coordinates.x1,
            "y1": chunk.coordinates.y1,
            "x2": chunk.coordinates.x2,
            "y2": chunk.coordinates.y2,
        },
        "metadata": dict(chunk.metadata),
        "image_path": chunk.image_path,
        "parent_id": chunk.parent_id,
        "chunk_level": chunk.chunk_level,
        "confidence": chunk.confidence,
        "needs_review": chunk.needs_review,
    }


def sse_event(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def text_stream_chunks(text: str, size: int = 36) -> Iterable[str]:
    for start in range(0, len(text), max(1, size)):
        yield text[start : start + size]


def auth_enforcement_enabled() -> bool:
    return os.getenv("RAG_AUTH_ENFORCEMENT_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def _legacy_data_endpoint(path: str) -> bool:
    return (
        path in {"/answer", "/answer/stream", "/vector/status", "/vector/collection", "/vector/ingest", "/vector/search"}
        or path.startswith("/kb/")
        or (path.startswith("/workspaces/") and path.count("/") == 2)
    )


def create_app(initial_chunks: Optional[Iterable[DocumentChunk]] = None, config: RAGConfig = DEFAULT_CONFIG) -> FastAPI:
    kb_state = KnowledgeBaseState(config=config, initial_chunks=initial_chunks)
    document_delete_jobs: Dict[str, Dict[str, Any]] = {}
    document_delete_jobs_lock = RLock()
    workspace_operation_locks: Dict[str, Lock] = {}
    workspace_operation_locks_lock = RLock()
    app = FastAPI(
        title="Education Multimodal RAG API",
        description="Source-grounded API for education-material multimodal RAG.",
        version="0.2.0",
    )
    app.state.kb = kb_state
    app.state.document_delete_jobs = document_delete_jobs
    app.state.document_delete_jobs_lock = document_delete_jobs_lock
    app.state.workspace_operation_locks = workspace_operation_locks
    app.state.auth_store = None

    def workspace_operation_lock(workspace_id: str) -> Lock:
        with workspace_operation_locks_lock:
            lock = workspace_operation_locks.get(workspace_id)
            if lock is None:
                lock = Lock()
                workspace_operation_locks[workspace_id] = lock
            return lock

    app.state.workspace_operation_lock = workspace_operation_lock

    def delete_workspace_data(workspace_id: str) -> Dict[str, Any]:
        if not workspace_id or len(workspace_id) > 128:
            raise HTTPException(status_code=400, detail="工作区 ID 无效")
        if workspace_id == DEFAULT_WORKSPACE_ID:
            raise HTTPException(status_code=400, detail="默认工作区不能删除")

        chunk_ids = kb_state.workspace_chunk_ids(workspace_id)
        if chunk_ids and config.vector_store_provider != "memory":
            try:
                vector_store = MilvusVectorStore(config=config, embedding=HashEmbeddingAdapter(config))
                vector_store.delete_chunk_ids(chunk_ids)
            except Exception as exc:  # noqa: BLE001 - retain workspace data if vector cleanup fails.
                write_debug_log(
                    config,
                    "workspace_delete_vector_error",
                    workspace_id=workspace_id,
                    error_type=type(exc).__name__,
                    error=sanitize_error_detail(str(exc)),
                )
                raise HTTPException(
                    status_code=503,
                    detail="工作区向量索引删除失败，请检查向量数据库连接后重试",
                ) from exc

        return kb_state.delete_workspace(workspace_id)

    app.state.delete_workspace_data = delete_workspace_data
    cors_origins = parse_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials="*" not in cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router, prefix="/api/auth")
    app.include_router(conversation_router)
    app.include_router(workspace_router)
    app.include_router(workspace_admin_router)

    async def authorize_legacy_request(request: Request) -> Optional[Response]:
        path = request.url.path
        if request.method == "OPTIONS":
            # Let CORSMiddleware answer browser preflight before requiring a session.
            return None
        if not _legacy_data_endpoint(path):
            return None
        enforcement_enabled = auth_enforcement_enabled()
        if not enforcement_enabled and path not in {"/answer", "/answer/stream"}:
            return None

        try:
            store = _auth_store(request)
            principal = await run_in_threadpool(store.current_principal, request.cookies.get("kujie_session"))
        except AuthServiceUnavailable:
            return JSONResponse(status_code=503, content={"detail": "账号服务暂不可用"})
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})
        if principal is None:
            if enforcement_enabled:
                return JSONResponse(status_code=401, content={"detail": "请先登录"})
            return None
        request.state.auth_principal = principal
        is_platform_admin = principal.user.get("platform_role") == "platform_admin"
        workspace_store = WorkspaceStore(store)

        def denied(status_code: int, detail: str) -> JSONResponse:
            return JSONResponse(status_code=status_code, content={"detail": detail})

        global_admin_paths = {"/vector/status", "/vector/collection"}
        is_legacy_workspace_delete = request.method == "DELETE" and path.startswith("/workspaces/")
        if path in global_admin_paths or is_legacy_workspace_delete:
            if not is_platform_admin:
                return denied(403, "需要平台管理员权限")

        is_write = request.method in {"POST", "PUT", "PATCH", "DELETE"}
        if is_write:
            try:
                _require_browser_request(request, require_json=request.method != "DELETE")
                _require_csrf(request, principal)
            except HTTPException as exc:
                return denied(exc.status_code, str(exc.detail))

        payload: dict[str, Any] = {}
        if request.method in {"POST", "PUT", "PATCH"}:
            raw_body = await request.body()
            try:
                parsed = json.loads(raw_body) if raw_body else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                return denied(400, "请求体不是有效的 JSON")
            if not isinstance(parsed, dict):
                return denied(400, "请求体格式无效")
            payload = parsed


        if path in {"/answer", "/answer/stream"}:
            scope = str(payload.get("scope") or "workspace")
            workspace_id = str(payload.get("workspace_id") or DEFAULT_WORKSPACE_ID)
            try:
                resolved = await run_in_threadpool(
                    workspace_store.resolve_readable_scope,
                    principal.user["id"],
                    scope,
                    workspace_id,
                )
            except WorkspaceNotFound as exc:
                return denied(404, str(exc))
            except WorkspaceForbidden as exc:
                return denied(403, str(exc))
            except ValueError as exc:
                return denied(422, str(exc))
            except AuthServiceUnavailable:
                return denied(503, "账号或工作区权限服务暂不可用")
            request.state.auth_workspace_scope = {**resolved, "scope": scope}
            request.state.auth_requested_workspace_id = workspace_id
        elif path == "/vector/ingest":
            workspace_id = str(payload.get("workspace_id") or "").strip()
            mode = str(payload.get("mode") or "append").strip().lower()
            if mode == "replace":
                if workspace_id:
                    return denied(422, "向量索引替换仅支持平台级操作")
                if not is_platform_admin:
                    return denied(403, "全量重建向量索引需要平台管理员权限")
            elif workspace_id:
                try:
                    access = await run_in_threadpool(
                        workspace_store.get_workspace,
                        principal.user["id"],
                        workspace_id,
                    )
                except WorkspaceNotFound as exc:
                    return denied(404, str(exc))
                except WorkspaceForbidden as exc:
                    return denied(403, str(exc))
                except AuthServiceUnavailable:
                    return denied(503, "账号或工作区权限服务暂不可用")
                if not access.get("can_manage_content"):
                    return denied(403, "需要本工作区管理权限")
                request.state.auth_workspace_id = workspace_id
            elif not is_platform_admin:
                return denied(403, "全库向量同步需要平台管理员权限")
        elif path in global_admin_paths or is_legacy_workspace_delete:
            # The platform administrator may run global maintenance operations.
            pass
        elif path == "/kb/documents/delete" or path.startswith("/kb/documents/delete/"):
            if request.method == "GET":
                job_id = path.rsplit("/", 1)[-1]
                with document_delete_jobs_lock:
                    job = document_delete_jobs.get(job_id)
                    job_snapshot = dict(job) if job else None
                if job_snapshot is None:
                    return denied(404, "删除任务不存在或已过期")
                if job_snapshot.get("created_by_user_id") != principal.user["id"]:
                    try:
                        access = await run_in_threadpool(
                            workspace_store.get_workspace,
                            principal.user["id"],
                            str(job_snapshot.get("workspace_id") or ""),
                        )
                    except (WorkspaceNotFound, WorkspaceForbidden):
                        return denied(404, "删除任务不存在或已过期")
                    if not access.get("can_manage_content"):
                        return denied(404, "删除任务不存在或已过期")
                request.state.auth_workspace_id = str(job_snapshot.get("workspace_id") or "")
            else:
                workspace_id = str(payload.get("workspace_id") or DEFAULT_WORKSPACE_ID)
                try:
                    access = await run_in_threadpool(
                        workspace_store.get_workspace,
                        principal.user["id"],
                        workspace_id,
                    )
                except WorkspaceNotFound as exc:
                    return denied(404, str(exc))
                except WorkspaceForbidden as exc:
                    return denied(403, str(exc))
                except AuthServiceUnavailable:
                    return denied(503, "账号或工作区权限服务暂不可用")
                if not access.get("can_manage_content"):
                    return denied(403, "需要本工作区管理权限")
                request.state.auth_workspace_id = workspace_id
        else:
            workspace_id = str(
                payload.get("workspace_id")
                or request.query_params.get("workspace_id")
                or DEFAULT_WORKSPACE_ID
            )
            try:
                access = await run_in_threadpool(
                    workspace_store.get_workspace,
                    principal.user["id"],
                    workspace_id,
                )
            except WorkspaceNotFound as exc:
                return denied(404, str(exc))
            except WorkspaceForbidden as exc:
                return denied(403, str(exc))
            except AuthServiceUnavailable:
                return denied(503, "账号或工作区权限服务暂不可用")
            if is_write and not access.get("can_manage_content"):
                return denied(403, "需要本工作区管理权限")
            if not is_write and not access.get("can_read"):
                return denied(403, "无权读取该工作区")
            request.state.auth_workspace_id = workspace_id
            request.state.auth_permission_cache_scope = (
                f"u{principal.user.get('permissions_version', 0)}:"
                f"{workspace_id}={access.get('version')}:{access.get('role')}"
            )

        return None

    async def answer_scope_change(request: Request) -> str | None:
        original_scope = getattr(request.state, "auth_workspace_scope", None)
        if original_scope is None:
            return None
        try:
            store = _auth_store(request)
            current = await run_in_threadpool(store.current_principal, request.cookies.get("kujie_session"))
            if current is None or current.user.get("id") != original_scope.get("user_id"):
                return "session_revoked"
            if current.user.get("permissions_version") != original_scope.get("user_permissions_version"):
                return "permission_changed"
            resolved = await run_in_threadpool(
                WorkspaceStore(store).resolve_readable_scope,
                current.user["id"],
                str(original_scope.get("scope") or "workspace"),
                request.state.auth_requested_workspace_id,
            )
            if resolved.get("cache_scope") != original_scope.get("cache_scope"):
                return "permission_changed"
            return None
        except AuthServiceUnavailable:
            return "permission_changed"
        except (WorkspaceForbidden, WorkspaceNotFound, ValueError):
            return "permission_changed"

    @app.middleware("http")
    async def preserve_error_reason_for_browser(request: Request, call_next):
        """Keep API errors readable by browsers from configured frontend origins."""

        operation_lock = None
        try:
            auth_response = await authorize_legacy_request(request)
            if auth_response is None:
                workspace_id = getattr(request.state, "auth_workspace_id", None)
                if workspace_id and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
                    operation_lock = workspace_operation_lock(str(workspace_id))
                    await run_in_threadpool(operation_lock.acquire)
                try:
                    response = await call_next(request)
                finally:
                    if operation_lock is not None:
                        operation_lock.release()
            else:
                response = auth_response
        except HTTPException as exc:
            response = JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})
        except Exception as exc:  # noqa: BLE001 - final API safety net.
            error = explain_api_error(exc, 500, type(exc).__name__)
            write_debug_log(config, "unhandled_api_error", error_type=type(exc).__name__, error=error["technical_detail"])
            response = JSONResponse(
                status_code=500,
                content={"detail": f"原因：{error['reason']} 处理建议：{error['suggestion']}", "error": error},
            )

        origin = request.headers.get("origin", "")
        if origin and (origin in cors_origins or "*" in cors_origins):
            response.headers.setdefault("Access-Control-Allow-Origin", origin)
            vary = response.headers.get("Vary", "")
            if "origin" not in {part.strip().lower() for part in vary.split(",")}:
                response.headers["Vary"] = f"{vary}, Origin" if vary else "Origin"
        return response

    @app.exception_handler(HTTPException)
    async def explain_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        error = explain_api_error(exc.detail, exc.status_code, type(exc).__name__)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": f"原因：{error['reason']} 处理建议：{error['suggestion']}", "error": error},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def explain_validation_exception(_request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = []
        for item in exc.errors():
            location = ".".join(str(part) for part in item.get("loc", []) if part != "body")
            message = str(item.get("msg") or "参数无效")
            fields.append(f"{location or '请求体'}: {message}")
        error = explain_api_error("；".join(fields), 422, type(exc).__name__)
        return JSONResponse(
            status_code=422,
            content={"detail": f"原因：{error['reason']} 处理建议：{error['suggestion']}", "error": error},
        )

    @app.exception_handler(Exception)
    async def explain_unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
        error = explain_api_error(exc, 500, type(exc).__name__)
        write_debug_log(config, "unhandled_api_error", error_type=type(exc).__name__, error=error["technical_detail"])
        return JSONResponse(
            status_code=500,
            content={"detail": f"原因：{error['reason']} 处理建议：{error['suggestion']}", "error": error},
        )

    if FRONTEND_DIR.exists():
        app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    def record_audit_event(
        actor_user_id: str | None,
        action: str,
        target_type: str,
        target_id: str | None,
        details: Dict[str, Any],
    ) -> None:
        if not actor_user_id:
            return
        store = getattr(app.state, "auth_store", None)
        if store is None:
            return
        try:
            store.record_audit_event(actor_user_id, action, target_type, target_id, details)
        except AuthServiceUnavailable as exc:
            write_debug_log(
                config,
                "auth_audit_write_failed",
                action=action,
                target_type=target_type,
                error_type=type(exc).__name__,
            )

    def record_request_workspace_audit(http_request: Request, action: str, details: Dict[str, Any]) -> None:
        principal = getattr(http_request.state, "auth_principal", None)
        if principal is None:
            return
        workspace_id = str(getattr(http_request.state, "auth_workspace_id", DEFAULT_WORKSPACE_ID))
        record_audit_event(principal.user["id"], action, "workspace", workspace_id, details)

    def update_document_delete_job(job_id: str, **updates: Any) -> None:
        with document_delete_jobs_lock:
            job = document_delete_jobs.get(job_id)
            if job is not None:
                job.update(updates)

    def run_document_delete_job(job_id: str, source: str, workspace_id: str, actor_user_id: str | None) -> None:
        try:
            update_document_delete_job(job_id, status="running", progress=5, phase="正在检查资料索引")
            chunk_ids = kb_state.document_chunk_ids(source, workspace_id)
            if not chunk_ids:
                raise HTTPException(status_code=404, detail=f"知识库中不存在资料：{source}")

            if config.vector_store_provider == "memory":
                update_document_delete_job(job_id, progress=78, phase="正在更新本地向量索引")
            else:
                update_document_delete_job(job_id, progress=12, phase="正在连接本地向量库")
                vector_store = MilvusVectorStore(config=config, embedding=HashEmbeddingAdapter(config))

                def report_vector_progress(done: int, total: int) -> None:
                    ratio = done / max(total, 1)
                    update_document_delete_job(
                        job_id,
                        progress=20 + round(ratio * 62),
                        phase=f"正在删除向量索引（{done}/{total}）",
                    )

                vector_store.delete_chunk_ids(chunk_ids, progress_callback=report_vector_progress)

            update_document_delete_job(job_id, progress=88, phase="正在从知识库移除资料")
            result = kb_state.delete_document(source, chunk_ids, workspace_id)
            record_audit_event(
                actor_user_id,
                "workspace_document_deleted",
                "workspace",
                workspace_id,
                {"source": source, "removed_chunk_count": result["removed_chunk_count"]},
            )
            update_document_delete_job(
                job_id,
                status="completed",
                progress=100,
                phase="删除完成",
                removed_chunk_count=result["removed_chunk_count"],
                summary=result["summary"],
                error="",
            )
        except Exception as exc:  # noqa: BLE001 - keep job state pollable when backend operations fail.
            detail = exc.detail if isinstance(exc, HTTPException) else exc
            status_code = exc.status_code if isinstance(exc, HTTPException) else 503
            error = explain_api_error(detail, status_code, type(exc).__name__)["reason"]
            update_document_delete_job(job_id, status="failed", phase="删除失败", error=error)

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "service": "education-rag-api",
            "health": "/health",
            "answer": "/answer",
            "answer_stream": "/answer/stream",
            "document_ingest": "/kb/document",
            "frontend": "/frontend/" if FRONTEND_DIR.exists() else None,
        }

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        infrastructure_status = kb_state.infrastructure.status()
        return HealthResponse(
            status="ok",
            service="education-rag-api",
            kb_chunk_count=kb_state.chunk_count,
            model_mode=f"{config.llm_provider}:{config.llm_model}",
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            rerank_provider=config.rerank_provider,
            rerank_model=config.rerank_model,
            ocr_provider=config.ocr_provider,
            vector_store_provider=config.vector_store_provider,
            vector_collection_name=config.zilliz_collection_name,
            strict_source_required=config.strict_source_required,
            default_kb_path=str(DEFAULT_KB_PATH),
            query_classifier=f"{config.query_classifier_provider}:{config.query_classifier_model}",
            query_agent=config.query_agent_provider,
            redis_status=str(infrastructure_status["redis"]),
            mysql_status=str(infrastructure_status["mysql"]),
        )

    @app.post("/answer")
    async def answer(payload: AnswerRequest, http_request: Request) -> Dict[str, Any]:
        scope = getattr(http_request.state, "auth_workspace_scope", None)
        principal = getattr(http_request.state, "auth_principal", None)
        answer_options: dict[str, Any] = {}
        if scope is not None:
            answer_options = {
                "workspace_ids": scope["workspace_ids"],
                "principal_user_id": scope["user_id"],
                "permission_cache_scope": scope["cache_scope"],
                "strict_workspace_metadata": True,
            }
        session_id = payload.session_id
        conversation_context = payload.conversation_context
        conversation_store = None
        conversation_id = payload.conversation_id
        if scope is not None and principal is not None:
            if not conversation_id:
                raise HTTPException(status_code=400, detail="请先创建服务器对话")
            conversation_store = ConversationStore(_auth_store(http_request))
            try:
                conversation_context = await run_in_threadpool(
                    conversation_store.begin_turn,
                    scope["user_id"],
                    conversation_id,
                    payload.scope,
                    payload.workspace_id if payload.scope == "workspace" else None,
                    scope["workspace_ids"],
                    payload.question.strip(),
                )
            except ConversationNotFound as exc:
                raise HTTPException(status_code=404, detail="对话不存在") from exc
            except ConversationLocked as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if conversation_context is None:
                raise HTTPException(status_code=403, detail="资料权限已变更，请新建对话")
            session_id = authenticated_session_id(scope["user_id"], conversation_id)
        response = await run_in_threadpool(
            kb_state.answer,
            payload.question.strip(),
            session_id,
            payload.rerank_rules.dict(),
            conversation_context=conversation_context,
            hallucination_threshold=payload.hallucination_threshold,
            workspace_id=payload.workspace_id,
            **answer_options,
        )
        change = await answer_scope_change(http_request)
        if change == "session_revoked":
            raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录")
        if change == "permission_changed":
            raise HTTPException(status_code=403, detail="资料权限已变更，请重新提问")
        if conversation_store is not None and scope is not None and conversation_id:
            serialized = response.to_dict()
            try:
                await run_in_threadpool(
                    conversation_store.append_assistant,
                    scope["user_id"],
                    conversation_id,
                    str(serialized.get("answer") or ""),
                    serialized.get("sources") or [],
                )
            except ConversationNotFound as exc:
                raise HTTPException(status_code=404, detail="对话不存在") from exc
            except ConversationLocked as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            change = await answer_scope_change(http_request)
            if change == "session_revoked":
                raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录")
            if change == "permission_changed":
                raise HTTPException(status_code=403, detail="资料权限已变更，请重新提问")
        return response.to_dict()

    @app.post("/answer/stream")
    async def answer_stream(payload: AnswerRequest, http_request: Request) -> StreamingResponse:
        scope = getattr(http_request.state, "auth_workspace_scope", None)
        principal = getattr(http_request.state, "auth_principal", None)
        answer_options: dict[str, Any] = {}
        if scope is not None:
            answer_options = {
                "workspace_ids": scope["workspace_ids"],
                "principal_user_id": scope["user_id"],
                "permission_cache_scope": scope["cache_scope"],
                "strict_workspace_metadata": True,
            }
        session_id = payload.session_id
        conversation_context = payload.conversation_context
        conversation_store = None
        conversation_id = payload.conversation_id
        if scope is not None and principal is not None:
            if not conversation_id:
                raise HTTPException(status_code=400, detail="请先创建服务器对话")
            conversation_store = ConversationStore(_auth_store(http_request))
            try:
                conversation_context = await run_in_threadpool(
                    conversation_store.begin_turn,
                    scope["user_id"],
                    conversation_id,
                    payload.scope,
                    payload.workspace_id if payload.scope == "workspace" else None,
                    scope["workspace_ids"],
                    payload.question.strip(),
                )
            except ConversationNotFound as exc:
                raise HTTPException(status_code=404, detail="对话不存在") from exc
            except ConversationLocked as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if conversation_context is None:
                raise HTTPException(status_code=403, detail="资料权限已变更，请新建对话")
            session_id = authenticated_session_id(scope["user_id"], conversation_id)

        async def events() -> AsyncIterator[str]:
            yield sse_event("status", {"stage": "intent", "message": "正在识别问题类型…"})
            event_queue: Queue[tuple[str, Dict[str, Any]]] = Queue()
            streamed = False
            stop_stream = Event()

            def enqueue_token(token: str) -> None:
                if token and not stop_stream.is_set():
                    event_queue.put(("token", {"text": token}))

            def enqueue_status(message: str) -> None:
                if message and not stop_stream.is_set():
                    event_queue.put(("status", {"message": message}))

            worker = asyncio.create_task(
                run_in_threadpool(
                    kb_state.answer,
                    payload.question.strip(),
                    session_id,
                    payload.rerank_rules.dict(),
                    conversation_context=conversation_context,
                    on_token=enqueue_token,
                    hallucination_threshold=payload.hallucination_threshold,
                    on_status=enqueue_status,
                    workspace_id=payload.workspace_id,
                    **answer_options,
                )
            )
            loop = asyncio.get_running_loop()
            last_permission_check = loop.time()
            try:
                while not worker.done() or not event_queue.empty():
                    if scope is not None and loop.time() - last_permission_check >= 2.0:
                        last_permission_check = loop.time()
                        change = await answer_scope_change(http_request)
                        if change:
                            stop_stream.set()
                            while not event_queue.empty():
                                try:
                                    event_queue.get_nowait()
                                except Empty:
                                    break
                            worker.cancel()
                            await asyncio.gather(worker, return_exceptions=True)
                            yield sse_event(change, {"message": "登录状态或资料权限已变更"})
                            return
                    try:
                        event_name, event_payload = event_queue.get_nowait()
                    except Empty:
                        await asyncio.sleep(0.015)
                        continue
                    if event_name == "token":
                        streamed = True
                    yield sse_event(event_name, event_payload)
                response = await worker
                change = await answer_scope_change(http_request)
                if change:
                    yield sse_event(change, {"message": "登录状态或资料权限已变更"})
                    return
            except Exception as exc:  # noqa: BLE001 - keep failures visible in an SSE stream.
                error = explain_api_error(exc, 500, type(exc).__name__)
                write_debug_log(
                    config,
                    "answer_stream_error",
                    error_type=type(exc).__name__,
                    error=error["technical_detail"],
                )
                yield sse_event("error", {
                    "code": error["code"],
                    "reason": error["reason"],
                    "suggestion": error["suggestion"],
                })
                return
            metadata = response.to_dict()
            answer_text = str(metadata.pop("answer", ""))
            if conversation_store is not None and scope is not None and conversation_id:
                try:
                    await run_in_threadpool(
                        conversation_store.append_assistant,
                        scope["user_id"],
                        conversation_id,
                        answer_text,
                        metadata.get("sources") or [],
                    )
                except ConversationLocked:
                    yield sse_event("permission_changed", {"message": "资料权限已变更"})
                    return
                change = await answer_scope_change(http_request)
                if change:
                    yield sse_event(change, {"message": "登录状态或资料权限已变更"})
                    return
            yield sse_event("metadata", metadata)
            if not streamed:
                for token_chunk in text_stream_chunks(answer_text):
                    yield sse_event("token", {"text": token_chunk})
            yield sse_event("done", {"answer": answer_text, "cache_hit": response.cache_hit})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/kb/summary", response_model=KnowledgeBaseSummary)
    def kb_summary(workspace_id: str = DEFAULT_WORKSPACE_ID) -> KnowledgeBaseSummary:
        return kb_state.summary(workspace_id)

    @app.delete("/workspaces/{workspace_id}")
    def delete_workspace(workspace_id: str) -> Dict[str, Any]:
        raise HTTPException(status_code=410, detail="请使用工作区管理页面永久删除授权工作区")

    @app.get("/kb/documents")
    def list_uploaded_documents(workspace_id: str = DEFAULT_WORKSPACE_ID) -> Dict[str, Any]:
        return kb_state.list_original_documents(workspace_id)

    @app.get("/kb/documents/content")
    def get_document_content(
        source: str,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        max_chars: int = 150_000,
    ) -> Dict[str, Any]:
        return kb_state.document_content(source, workspace_id, max_chars)

    @app.get("/kb/documents/preview")
    def preview_document(source: str, workspace_id: str = DEFAULT_WORKSPACE_ID) -> FileResponse:
        original_path = stored_document_path(workspace_id, source)
        if not original_path.is_file():
            raise HTTPException(status_code=404, detail="原始文档不存在，请重新上传后预览")

        filename = stored_document_filename(workspace_id, source)
        return FileResponse(
            original_path,
            media_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
            headers={
                "Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}",
                "X-Original-Document": "true",
            },
        )

    @app.get("/kb/documents/download")
    def download_document(source: str, workspace_id: str = DEFAULT_WORKSPACE_ID) -> Response:
        document = kb_state.document_content(source, workspace_id, max_chars=0)
        original_path = stored_document_path(workspace_id, source)
        filename = stored_document_filename(workspace_id, source)
        if original_path.is_file():
            return FileResponse(
                original_path,
                media_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
                filename=filename,
                headers={"X-Original-Document": "true"},
            )
        export_name = filename if filename.lower().endswith(".txt") else f"{filename}.txt"
        return Response(
            content=document["content"].encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(export_name)}",
                "X-Original-Document": "false",
            },
        )

    @app.post("/kb/documents/delete")
    def begin_document_delete(
        request: DocumentDeleteRequest,
        background_tasks: BackgroundTasks,
        http_request: Request,
    ) -> Dict[str, Any]:
        source = request.source.strip()
        if not source:
            raise HTTPException(status_code=400, detail="资料名称不能为空")
        if not kb_state.document_chunk_ids(source, request.workspace_id):
            raise HTTPException(status_code=404, detail=f"知识库中不存在资料：{source}")
        job_id = uuid4().hex
        principal = getattr(http_request.state, "auth_principal", None)
        job = {
            "job_id": job_id,
            "source": source,
            "workspace_id": request.workspace_id,
            "created_by_user_id": principal.user["id"] if principal else None,
            "status": "queued",
            "progress": 0,
            "phase": "等待删除",
            "removed_chunk_count": 0,
            "error": "",
        }
        with document_delete_jobs_lock:
            if len(document_delete_jobs) >= 100:
                for old_job_id, old_job in list(document_delete_jobs.items()):
                    if old_job.get("status") in {"completed", "failed"}:
                        document_delete_jobs.pop(old_job_id, None)
                    if len(document_delete_jobs) < 90:
                        break
            document_delete_jobs[job_id] = job
        background_tasks.add_task(run_document_delete_job, job_id, source, request.workspace_id, job["created_by_user_id"])
        return dict(job)

    @app.get("/kb/documents/delete/{job_id}")
    def get_document_delete_progress(job_id: str) -> Dict[str, Any]:
        with document_delete_jobs_lock:
            job = document_delete_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
            return dict(job)

    @app.get("/kb/chunks")
    def list_kb_chunks(limit: int = 50, offset: int = 0, workspace_id: str = DEFAULT_WORKSPACE_ID) -> Dict[str, Any]:
        return kb_state.list_chunks(limit=limit, offset=offset, workspace_id=workspace_id)

    @app.post("/kb/chunks", response_model=KnowledgeBaseUpdateResponse)
    def update_kb_chunks(request: KnowledgeBaseReplaceRequest, http_request: Request) -> KnowledgeBaseUpdateResponse:
        result = kb_state.update_chunks(request.chunks, mode=request.mode, workspace_id=request.workspace_id)
        record_request_workspace_audit(http_request, "workspace_documents_updated", {"mode": result.mode, "chunk_count": result.chunk_count})
        return result

    @app.post("/kb/jsonl", response_model=KnowledgeBaseUpdateResponse)
    def ingest_kb_jsonl(request: JsonlIngestRequest, http_request: Request) -> KnowledgeBaseUpdateResponse:
        result = kb_state.ingest_jsonl(
            request.content,
            mode=request.mode,
            source_name=request.source_name,
            workspace_id=request.workspace_id,
        )
        record_request_workspace_audit(http_request, "workspace_document_ingested", {"source": safe_document_filename(request.source_name or "JSONL资料"), "mode": result.mode, "chunk_count": result.chunk_count})
        return result

    @app.post("/kb/pdf", response_model=KnowledgeBaseUpdateResponse)
    def ingest_kb_pdf(request: PdfIngestRequest, http_request: Request) -> KnowledgeBaseUpdateResponse:
        result = kb_state.ingest_pdf(
            request.filename,
            request.content_base64,
            mode=request.mode,
            max_pages=request.max_pages,
            render_dpi=request.render_dpi,
            workspace_id=request.workspace_id,
        )
        record_request_workspace_audit(http_request, "workspace_document_ingested", {"source": safe_document_filename(request.filename), "mode": result.mode, "chunk_count": result.chunk_count})
        return result

    @app.post("/kb/image-ocr", response_model=KnowledgeBaseUpdateResponse)
    def ingest_kb_image_ocr(request: ImageOcrIngestRequest, http_request: Request) -> KnowledgeBaseUpdateResponse:
        result = kb_state.ingest_image_ocr(
            request.filename,
            request.content_base64,
            ocr_text=request.ocr_text,
            mode=request.mode,
            workspace_id=request.workspace_id,
        )
        record_request_workspace_audit(http_request, "workspace_document_ingested", {"source": safe_document_filename(request.filename), "mode": result.mode, "chunk_count": result.chunk_count})
        return result

    @app.post("/kb/document", response_model=KnowledgeBaseUpdateResponse)
    def ingest_kb_document(request: DocumentIngestRequest, http_request: Request) -> KnowledgeBaseUpdateResponse:
        result = kb_state.ingest_document(
            request.filename,
            request.content_base64,
            mode=request.mode,
            max_pages=request.max_pages,
            render_dpi=request.render_dpi,
            workspace_id=request.workspace_id,
        )
        record_request_workspace_audit(http_request, "workspace_document_ingested", {"source": safe_document_filename(request.filename), "mode": result.mode, "chunk_count": result.chunk_count})
        return result

    @app.get("/vector/status")
    def vector_status() -> Dict[str, Any]:
        if config.vector_store_provider == "memory":
            return {
                "provider": "memory",
                "connected": True,
                "endpoint_configured": False,
                "token_configured": False,
                "collection_name": "in_memory_hybrid_index",
                "collection_exists": True,
                "row_count": kb_state.chunk_count,
                "message": "in-memory BM25 + dense/sparse index ready",
            }
        return MilvusVectorStore(config=config).status().to_dict()

    @app.post("/vector/collection")
    def create_vector_collection(request: VectorCollectionRequest) -> Dict[str, Any]:
        if config.vector_store_provider == "memory":
            return {
                "provider": "memory",
                "connected": True,
                "collection_name": "in_memory_hybrid_index",
                "collection_exists": True,
                "row_count": kb_state.chunk_count,
                "message": "memory mode has no remote collection to recreate",
            }
        try:
            return MilvusVectorStore(config=config).ensure_collection(drop_existing=request.drop_existing).to_dict()
        except Exception as exc:  # noqa: BLE001 - surface sanitized operational error to API client.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/vector/ingest")
    def ingest_vector_collection(request: VectorIngestRequest = VectorIngestRequest()) -> Dict[str, Any]:
        normalized_mode = validate_mode(request.mode)
        if normalized_mode == "replace" and request.workspace_id:
            raise HTTPException(status_code=422, detail="向量索引替换仅支持平台级操作")
        chunks = kb_state.chunks_snapshot(request.workspace_id) if request.workspace_id else kb_state.chunks_snapshot()
        if config.vector_store_provider == "memory":
            return {
                "status": "ok",
                "inserted": len(chunks),
                "collection_name": "in_memory_hybrid_index",
                "mode": normalized_mode,
                "provider": "memory",
            }
        try:
            store = MilvusVectorStore(config=config)
            if normalized_mode == "replace":
                log_chunks_for_ingest(config, "vector_replace_payload", chunks, mode=normalized_mode)
                result = store.replace_chunks(chunks)
                result_event = "vector_replace_result"
            else:
                log_chunks_for_ingest(config, "vector_append_payload", chunks, mode=normalized_mode)
                result = store.upsert_chunks(chunks)
                result["mode"] = normalized_mode
                result_event = "vector_append_result"
            write_debug_log(config, result_event, **result)
            return result
        except Exception as exc:  # noqa: BLE001 - surface sanitized operational error to API client.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/vector/search")
    def search_vector_collection(payload: VectorSearchRequest, http_request: Request) -> Dict[str, Any]:
        try:
            processor = AdaptiveQueryOptimizer(build_llm_router(config), config)
            intent = processor.classify_intent(payload.query)
            entities = processor.extract_entities(payload.query)
            query_variants = processor.expand_queries(payload.query, intent, entities)
            if not query_variants:
                query_variants = [payload.query]
            optimized_query = query_variants[1] if len(query_variants) > 1 else query_variants[0]
            workspace_chunks = kb_state.chunks_snapshot(payload.workspace_id)
            if config.vector_store_provider == "memory":
                retriever = HybridRetriever(
                    workspace_chunks,
                    embedding=build_embedding_adapter(config),
                    config=config,
                )
            else:
                local_embedding = (
                    HashEmbeddingAdapter(config)
                    if config.embedding_provider in {"bailian", "dashscope", "aliyun"}
                    else build_embedding_adapter(config)
                )
                retriever = MilvusHybridRetriever(
                    MilvusVectorStore(config=config),
                    workspace_chunks,
                    embedding=local_embedding,
                    config=config,
                    workspace_id=payload.workspace_id,
                    workspace_ids=(payload.workspace_id,) if auth_enforcement_enabled() else None,
                    strict_workspace_metadata=auth_enforcement_enabled(),
                )
            results = retrieve_with_query_variants(
                retriever,
                query_variants,
                entities,
                limit=payload.limit,
                rrf_k=config.rrf_k,
            )
            write_debug_log(
                config,
                "vector_search_results",
                query=payload.query,
                optimized_query=optimized_query,
                query_variants=query_variants,
                result_count=len(results),
                results=retrieval_results_for_debug(results, text_chars=config.debug_log_text_chars),
            )
            return {
                "query": payload.query,
                "optimized_query": optimized_query,
                "query_variants": query_variants,
                "entities": entities.to_dict(),
                "results": [
                    {
                        "rank": result.rank,
                        "score": result.score,
                        "source": result.chunk.to_source_ref(score=result.score).to_dict(),
                        "text": result.chunk.text,
                    }
                    for result in results
                ],
            }
        except Exception as exc:  # noqa: BLE001 - surface sanitized operational error to API client.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def parse_cors_origins() -> List[str]:
    configured = os.getenv("RAG_CORS_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return ["*"]


app = create_app()
