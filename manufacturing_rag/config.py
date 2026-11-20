"""Runtime configuration for the lightweight four-model RAG workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def _load_project_env() -> None:
    """Load the project .env for local runs without overriding shell variables."""

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name or os.environ.get(name, "").strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ[name] = value


_load_project_env()


@dataclass(frozen=True)
class RAGConfig:
    """Central knobs shared by ETL, adaptive retrieval and service infrastructure."""

    top_k: int = 5
    candidate_k: int = 20
    rerank_threshold: float = 0.08
    min_context_count: int = 1
    max_query_rewrites: int = 2
    rrf_k: int = 60
    retry_attempts: int = 3
    retry_base_seconds: float = 0.35
    parent_chunk_chars: int = 1800
    child_chunk_chars: int = 520
    child_chunk_overlap: int = 80
    min_etl_confidence: float = 0.65
    bm25_weight: float = 0.30
    sparse_weight: float = 0.25
    dense_weight: float = 0.45
    mmr_lambda: float = 0.72
    llm_temperature: float = 0.1
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-v4-flash"
    llm_api_base_url: str = "https://api.deepseek.com/chat/completions"
    llm_api_key_env: str = "DEEPSEEK_API_KEY"
    llm_timeout_seconds: float = 45.0
    llm_max_tokens: int = 1200
    llm_enable_thinking: bool = False
    llm_fallback_enabled: bool = True
    agent_loop_enabled: bool = True
    agent_request_timeout_seconds: float = 180.0
    embedding_provider: str = "bailian"
    embedding_model: str = "qwen3-vl-embedding"
    embedding_device: str = "cpu"
    embedding_api_base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    embedding_api_key_env: str = "BAILIAN_API_KEY"
    embedding_request_format: str = "dashscope_multimodal"
    embedding_timeout_seconds: float = 45.0
    embedding_batch_size: int = 16
    embedding_fallback_enabled: bool = False
    rerank_provider: str = "bailian"
    rerank_model: str = "qwen3-vl-rerank"
    rerank_device: str = "cpu"
    rerank_api_base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    rerank_api_key_env: str = "BAILIAN_API_KEY"
    rerank_timeout_seconds: float = 45.0
    rerank_fallback_enabled: bool = True
    ocr_provider: str = "paddleocr"
    paddle_ocr_language: str = "ch"
    paddle_ocr_use_gpu: bool = False
    baidu_ocr_api_key_env: str = "BAIDU_OCR_API_KEY"
    baidu_ocr_secret_key_env: str = "BAIDU_OCR_SECRET_KEY"
    baidu_ocr_token_url: str = "https://aip.baidubce.com/oauth/2.0/token"
    baidu_ocr_endpoint: str = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    baidu_ocr_timeout_seconds: float = 45.0
    baidu_ocr_pdf_max_pages: int = 0
    baidu_ocr_pdf_render_dpi: int = 150
    vision_provider: str = "qwen-vl"
    vision_model: str = "qwen3-vl-plus"
    vision_api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    vision_api_key_env: str = "BAILIAN_API_KEY"
    vision_timeout_seconds: float = 60.0
    vision_fallback_enabled: bool = True
    query_classifier_provider: str = "bert"
    query_classifier_model: str = "models/query-intent-bert"
    query_classifier_device: int = -1
    query_agent_provider: str = "langchain"
    vector_store_provider: str = "milvus"
    kb_runtime_path: str = "tmp/runtime_knowledge_base.jsonl"
    # Field names remain compatible with older integrations; runtime defaults
    # are local Docker Milvus variables and do not target Zilliz Cloud.
    zilliz_endpoint_env: str = "MILVUS_ENDPOINT"
    zilliz_token_env: str = "MILVUS_TOKEN"
    zilliz_collection_name: str = "education_multimodal_qwen3_vl"
    zilliz_alias: str = "milvus_default"
    zilliz_metric_type: str = "COSINE"
    zilliz_index_type: str = "AUTOINDEX"
    zilliz_consistency_level: str = "Bounded"
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = False
    redis_cache_ttl_seconds: int = 900
    redis_session_ttl_seconds: int = 86400
    mysql_url: str = "mysql+pymysql://rag:rag@localhost:3306/education_rag?charset=utf8mb4"
    mysql_enabled: bool = False
    strict_source_required: bool = True
    max_answer_chars_per_source: int = 420
    embedding_dimensions: int = 1024
    debug_log_enabled: bool = True
    debug_log_path: str = "logs/rag_debug.jsonl"
    debug_log_text_chars: int = 800
    source_label_prefix: str = "资料来源"
    sensitive_terms: Tuple[str, ...] = field(
        default=(
            "绝密",
            "机密",
            "商业秘密",
            "受控文件",
            "内部题库",
            "泄题",
            "学生隐私",
            "成绩单",
            "未授权",
            "密码",
            "密钥",
            "token",
            "api_key",
            "apikey",
            "secret",
        )
    )


def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""

    env_path = os.path.abspath(path)
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _bailian_url(path: str, fallback: str) -> str:
    """Build a Beijing workspace endpoint without storing account identifiers."""

    host = _env_str("BAILIAN_API_HOST", "")
    if not host:
        workspace_id = _env_str("BAILIAN_WORKSPACE_ID", "")
        host = f"{workspace_id}.cn-beijing.maas.aliyuncs.com" if workspace_id else ""
    if not host:
        return fallback
    host = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    return f"https://{host}{path}"


def config_from_env() -> RAGConfig:
    """Build runtime config from environment variables without exposing secrets."""

    return RAGConfig(
        top_k=_env_int("RAG_TOP_K", 5),
        candidate_k=_env_int("RAG_CANDIDATE_K", 20),
        rerank_threshold=_env_float("RAG_RERANK_THRESHOLD", 0.08),
        min_context_count=_env_int("RAG_MIN_CONTEXT_COUNT", 1),
        max_query_rewrites=_env_int("RAG_MAX_QUERY_REWRITES", 2),
        rrf_k=_env_int("RAG_RRF_K", 60),
        retry_attempts=_env_int("RAG_RETRY_ATTEMPTS", 3),
        retry_base_seconds=_env_float("RAG_RETRY_BASE_SECONDS", 0.35),
        parent_chunk_chars=_env_int("RAG_PARENT_CHUNK_CHARS", 1800),
        child_chunk_chars=_env_int("RAG_CHILD_CHUNK_CHARS", 520),
        child_chunk_overlap=_env_int("RAG_CHILD_CHUNK_OVERLAP", 80),
        min_etl_confidence=_env_float("RAG_MIN_ETL_CONFIDENCE", 0.65),
        bm25_weight=_env_float("RAG_BM25_WEIGHT", 0.30),
        sparse_weight=_env_float("RAG_SPARSE_WEIGHT", 0.25),
        dense_weight=_env_float("RAG_DENSE_WEIGHT", 0.45),
        mmr_lambda=_env_float("RAG_MMR_LAMBDA", 0.72),
        llm_temperature=_env_float("RAG_LLM_TEMPERATURE", 0.1),
        llm_provider=_env_str("RAG_LLM_PROVIDER", "deepseek").lower(),
        llm_model=_env_str("RAG_LLM_MODEL", "deepseek-v4-flash"),
        llm_api_base_url=_env_str(
            "RAG_LLM_API_BASE_URL",
            _env_str(
                "DEEPSEEK_API_BASE_URL",
                "https://api.deepseek.com/chat/completions",
            ),
        ),
        llm_api_key_env=_env_str(
            "RAG_LLM_API_KEY_ENV",
            _env_str("DEEPSEEK_API_KEY_ENV", "DEEPSEEK_API_KEY"),
        ),
        llm_timeout_seconds=_env_float("RAG_LLM_TIMEOUT_SECONDS", 45.0),
        llm_max_tokens=_env_int("RAG_LLM_MAX_TOKENS", 1200),
        llm_enable_thinking=_env_bool("RAG_LLM_ENABLE_THINKING", False),
        llm_fallback_enabled=_env_bool("RAG_LLM_FALLBACK_ENABLED", True),
        agent_loop_enabled=_env_bool("RAG_AGENT_LOOP_ENABLED", True),
        agent_request_timeout_seconds=_env_float("RAG_AGENT_REQUEST_TIMEOUT_SECONDS", 180.0),
        embedding_provider=_env_str("RAG_EMBEDDING_PROVIDER", "bailian").lower(),
        embedding_model=_env_str("RAG_EMBEDDING_MODEL", _env_str("BAILIAN_EMBEDDING_MODEL", "qwen3-vl-embedding")),
        embedding_device=_env_str("RAG_EMBEDDING_DEVICE", "cpu"),
        embedding_api_base_url=_env_str(
            "BAILIAN_EMBEDDING_API_BASE_URL",
            _bailian_url(
                "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
                "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
            ),
        ),
        embedding_api_key_env=_env_str("BAILIAN_API_KEY_ENV", "BAILIAN_API_KEY"),
        embedding_request_format=_env_str("BAILIAN_EMBEDDING_REQUEST_FORMAT", "dashscope_multimodal").lower(),
        embedding_timeout_seconds=_env_float("RAG_EMBEDDING_TIMEOUT_SECONDS", 45.0),
        embedding_batch_size=_env_int("RAG_EMBEDDING_BATCH_SIZE", 16),
        embedding_fallback_enabled=_env_bool("RAG_EMBEDDING_FALLBACK_ENABLED", False),
        rerank_provider=_env_str("RAG_RERANK_PROVIDER", "bailian").lower(),
        rerank_model=_env_str("RAG_RERANK_MODEL", _env_str("BAILIAN_RERANK_MODEL", "qwen3-vl-rerank")),
        rerank_device=_env_str("RAG_RERANK_DEVICE", "cpu"),
        rerank_api_base_url=_env_str(
            "BAILIAN_RERANK_API_BASE_URL",
            _bailian_url(
                "/api/v1/services/rerank/text-rerank/text-rerank",
                "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            ),
        ),
        rerank_api_key_env=_env_str("BAILIAN_API_KEY_ENV", "BAILIAN_API_KEY"),
        rerank_timeout_seconds=_env_float("RAG_RERANK_TIMEOUT_SECONDS", 45.0),
        rerank_fallback_enabled=_env_bool("RAG_RERANK_FALLBACK_ENABLED", True),
        ocr_provider=_env_str("RAG_OCR_PROVIDER", "paddleocr").lower(),
        paddle_ocr_language=_env_str("PADDLE_OCR_LANGUAGE", "ch"),
        paddle_ocr_use_gpu=_env_bool("PADDLE_OCR_USE_GPU", False),
        baidu_ocr_api_key_env=_env_str("BAIDU_OCR_API_KEY_ENV", "BAIDU_OCR_API_KEY"),
        baidu_ocr_secret_key_env=_env_str("BAIDU_OCR_SECRET_KEY_ENV", "BAIDU_OCR_SECRET_KEY"),
        baidu_ocr_token_url=_env_str("BAIDU_OCR_TOKEN_URL", "https://aip.baidubce.com/oauth/2.0/token"),
        baidu_ocr_endpoint=_env_str("BAIDU_OCR_ENDPOINT", "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"),
        baidu_ocr_timeout_seconds=_env_float("BAIDU_OCR_TIMEOUT_SECONDS", 45.0),
        baidu_ocr_pdf_max_pages=_env_int("BAIDU_OCR_PDF_MAX_PAGES", 0),
        baidu_ocr_pdf_render_dpi=_env_int("BAIDU_OCR_PDF_RENDER_DPI", 150),
        vision_provider=_env_str("RAG_VISION_PROVIDER", "qwen-vl").lower(),
        vision_model=_env_str("QWEN_VL_MODEL", "qwen3-vl-plus"),
        vision_api_base_url=_env_str(
            "QWEN_VL_API_BASE_URL",
            _bailian_url(
                "/compatible-mode/v1/chat/completions",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            ),
        ),
        vision_api_key_env=_env_str("RAG_VISION_API_KEY_ENV", "BAILIAN_API_KEY"),
        vision_timeout_seconds=_env_float("QWEN_VL_TIMEOUT_SECONDS", 60.0),
        vision_fallback_enabled=_env_bool("RAG_VISION_FALLBACK_ENABLED", _env_bool("QWEN_VL_FALLBACK_ENABLED", True)),
        query_classifier_provider=_env_str("RAG_QUERY_CLASSIFIER_PROVIDER", "bert").lower(),
        query_classifier_model=_env_str("RAG_BERT_MODEL", "models/query-intent-bert"),
        query_classifier_device=_env_int("RAG_BERT_DEVICE", -1),
        query_agent_provider=_env_str("RAG_QUERY_AGENT_PROVIDER", "langchain").lower(),
        vector_store_provider=_env_str("RAG_VECTOR_STORE_PROVIDER", "milvus").lower(),
        kb_runtime_path=_env_str("RAG_KB_RUNTIME_PATH", "tmp/runtime_knowledge_base.jsonl"),
        zilliz_endpoint_env=_env_str("MILVUS_ENDPOINT_ENV", "MILVUS_ENDPOINT"),
        zilliz_token_env=_env_str("MILVUS_TOKEN_ENV", "MILVUS_TOKEN"),
        zilliz_collection_name=_env_str("MILVUS_COLLECTION_NAME", "education_multimodal_qwen3_vl"),
        zilliz_alias=_env_str("MILVUS_ALIAS", "milvus_default"),
        zilliz_metric_type=_env_str("MILVUS_METRIC_TYPE", "COSINE"),
        zilliz_index_type=_env_str("MILVUS_INDEX_TYPE", "AUTOINDEX"),
        zilliz_consistency_level=_env_str("MILVUS_CONSISTENCY_LEVEL", "Bounded"),
        redis_url=_env_str("REDIS_URL", "redis://localhost:6379/0"),
        redis_enabled=_env_bool("RAG_REDIS_ENABLED", False),
        redis_cache_ttl_seconds=_env_int("RAG_REDIS_CACHE_TTL_SECONDS", 900),
        redis_session_ttl_seconds=_env_int("RAG_REDIS_SESSION_TTL_SECONDS", 86400),
        mysql_url=_env_str("MYSQL_URL", "mysql+pymysql://rag:rag@localhost:3306/education_rag?charset=utf8mb4"),
        mysql_enabled=_env_bool("RAG_MYSQL_ENABLED", False),
        strict_source_required=_env_bool("RAG_STRICT_SOURCE_REQUIRED", True),
        max_answer_chars_per_source=_env_int("RAG_MAX_ANSWER_CHARS_PER_SOURCE", 420),
        embedding_dimensions=_env_int("RAG_EMBEDDING_DIMENSIONS", 1024),
        debug_log_enabled=_env_bool("RAG_DEBUG_LOG_ENABLED", True),
        debug_log_path=_env_str("RAG_DEBUG_LOG_PATH", "logs/rag_debug.jsonl"),
        debug_log_text_chars=_env_int("RAG_DEBUG_LOG_TEXT_CHARS", 800),
    )


load_env_file()
DEFAULT_CONFIG = config_from_env()
