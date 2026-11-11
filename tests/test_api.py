import base64
import json
from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from manufacturing_rag.api import KnowledgeBaseState, create_app, pdf_chunks_are_searchable, read_pdf_literal_string
from manufacturing_rag.config import DEFAULT_CONFIG
from manufacturing_rag.schemas import DocumentChunk


def build_client(config=DEFAULT_CONFIG):
    chunks = [
        DocumentChunk(
            id="api-course-1",
            text="高等数学资料：极限的定义描述了当自变量趋近某一点或无穷远时，函数值无限接近某个确定常数的过程。",
            source="高等数学讲义OCR.jsonl",
            page=8,
            block_id="b02",
        )
    ]
    return TestClient(create_app(initial_chunks=chunks, config=config))


def test_document_delete_background_job_reaches_completed_status(monkeypatch, tmp_path):
    source = "delete-job-regression.pdf"
    chunks = [
        DocumentChunk(
            id="delete-job-regression-1",
            text="用于验证资料删除后台任务可以完成。",
            source=source,
            page=1,
            block_id="delete-1",
        )
    ]
    config = replace(
        DEFAULT_CONFIG,
        vector_store_provider="memory",
        kb_runtime_path=str(tmp_path / "runtime-kb.jsonl"),
    )
    monkeypatch.setattr("manufacturing_rag.api.remove_stored_document", lambda *_args, **_kwargs: None)
    client = TestClient(create_app(initial_chunks=chunks, config=config))

    started = client.post("/kb/documents/delete", json={"source": source})
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    completed = client.get(f"/kb/documents/delete/{job_id}")

    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["progress"] == 100
    assert completed.json()["phase"] == "删除完成"
    assert completed.json()["removed_chunk_count"] == 1


def test_health_endpoint_reports_kb_count():
    client = build_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["kb_chunk_count"] == 1


def test_answer_endpoint_returns_rag_response_shape():
    client = build_client()

    response = client.post("/answer", json={"question": "这份文档里极限的定义是什么？"})

    assert response.status_code == 200
    payload = response.json()
    assert "函数值" in payload["answer"]
    assert payload["sources"]
    assert payload["hallucination_check"]["passed"] is True


def test_jsonl_ingest_can_replace_kb():
    client = build_client()
    record = {
        "id": "api-assignment-1",
        "text": "习概资料：新时代中国特色社会主义思想强调坚持人民至上、自信自立、守正创新、问题导向、系统观念和胸怀天下。",
        "source": "习概讲义OCR.jsonl",
        "page": 2,
        "block_id": "b01",
        "metadata": {"doc_type": "学科讲义"},
    }

    response = client.post("/kb/jsonl", json={"content": json.dumps(record, ensure_ascii=False), "mode": "replace"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunk_count"] == 1
    assert payload["summary"]["doc_types"] == ["学科讲义"]


def test_jsonl_ingest_defaults_to_append_and_updates_same_source():
    client = build_client()
    first_record = {
        "id": "scholarship-1",
        "text": "奖学金制度第一版内容。",
        "source": "学生手册.pdf",
        "page": 10,
        "block_id": "p10-c1",
    }
    response = client.post("/kb/jsonl", json={"content": json.dumps(first_record, ensure_ascii=False)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "append"
    assert payload["chunk_count"] == 2
    assert set(payload["summary"]["sources"]) == {"高等数学讲义OCR.jsonl", "学生手册.pdf"}

    updated_record = {**first_record, "text": "奖学金制度更新后的内容。"}
    response = client.post(
        "/kb/jsonl",
        json={"content": json.dumps(updated_record, ensure_ascii=False), "mode": "append"},
    )

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 2
    chunks = client.get("/kb/chunks?limit=200").json()["chunks"]
    handbook_chunks = [chunk for chunk in chunks if chunk["source"] == "学生手册.pdf"]
    assert len(handbook_chunks) == 1
    assert handbook_chunks[0]["text"] == "奖学金制度更新后的内容。"


def test_uploaded_kb_is_restored_after_state_recreation(tmp_path):
    runtime_path = tmp_path / "runtime-kb.jsonl"
    config = replace(DEFAULT_CONFIG, vector_store_provider="memory", kb_runtime_path=str(runtime_path))
    state = KnowledgeBaseState(config=config)
    payload = {
        "id": "persistent-1",
        "text": "这条资料应在服务重启后仍然存在。",
        "source": "persistent.pdf",
        "page": 1,
    }

    response = state.update_chunks([payload], mode="append")
    restored = KnowledgeBaseState(config=config)

    assert response.chunk_count == restored.chunk_count
    assert any(chunk.source == "persistent.pdf" for chunk in restored.chunks_snapshot())


def test_pdf_ingest_can_replace_kb_from_text_pdf():
    client = build_client()
    pdf_bytes = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 71 >>
stream
BT /F1 12 Tf 72 720 Td (Course chapter function concept learning goal) Tj ET
endstream
endobj
%%EOF
"""

    response = client.post(
        "/kb/pdf",
        json={"filename": "lesson.pdf", "content_base64": base64.b64encode(pdf_bytes).decode("ascii"), "mode": "replace"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunk_count"] == 1
    assert payload["summary"]["sources"] == ["lesson.pdf"]


def test_pdf_ingest_uses_baidu_ocr_when_text_is_garbled(monkeypatch):
    client = build_client()
    monkeypatch.setattr(
        "manufacturing_rag.api.extract_pdf_text_by_page",
        lambda _pdf_bytes: ["é¡µ ä¸ é å® è¾ å· â¾ æ° å½ ä¸ é¡µ ä¸ ç¬ ã â¼¤"],
    )

    def fake_parse_pdf_bytes(self, pdf_bytes, source, max_pages=None, render_dpi=None):
        return [
            DocumentChunk(
                id=f"{source}:baidu-ocr:p1",
                text="MySQL 锁机制包括共享锁、排他锁、意向锁和行锁。",
                source=source,
                page=1,
                block_id="baidu-ocr-p1",
                modality="ocr_text",
                metadata={"doc_type": "PDF OCR", "ingest_format": "pdf_ocr", "ocr_provider": "baidu", "line_count": 1},
            )
        ]

    monkeypatch.setattr("manufacturing_rag.api.BaiduOCRAdapter.parse_pdf_bytes", fake_parse_pdf_bytes)
    pdf_bytes = b"%PDF-1.4\n%%EOF\n"

    response = client.post(
        "/kb/pdf",
        json={"filename": "mysql-locks.pdf", "content_base64": base64.b64encode(pdf_bytes).decode("ascii"), "mode": "replace"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunk_count"] == 1
    assert payload["summary"]["doc_types"] == ["PDF OCR"]


def test_pdf_searchable_detection_rejects_spaced_chinese_garble():
    garbled_text = " ".join(
        "页 不 锁 定 输 号 ⾏ 数 命 不 页 不 独 、 ⼤ 、 原 锁 等 不 页 不 了 级 部 置 向 尽 不 页 不 注 理 理 否 D 背 语 就 兼 需 不 页 不 耗 合 理 ⾼ 了 级 写 ⼊ 、 好 单 、 溃 ⽰ 、 功 理 系 统 不 ③ 不 ⻚ 锁 （ P a 删 意 不 L 否 有 k ⽌ 不 页 不 锁 定 号 ⻚ 数 命 （ 竞 y 独 合 L 语 就 1 法 K 背".split()
    )

    assert pdf_chunks_are_searchable([DocumentChunk(id="bad", text=garbled_text, source="MySQL锁基础知识.pdf")]) is False


def test_pdf_searchable_detection_accepts_normal_chinese_text():
    text = "MySQL 锁机制包括共享锁、排他锁、意向锁和行锁。乐观锁通过版本号或时间戳控制并发更新，适合读多写少场景。"

    assert pdf_chunks_are_searchable([DocumentChunk(id="ok", text=text, source="MySQL锁基础知识.pdf")]) is True


def test_pdf_literal_string_tolerates_out_of_range_octal_escape():
    value, next_index = read_pdf_literal_string(b"\\777)", 0)

    assert value == b"\xff"
    assert next_index == 5


def test_pdf_ingest_forwards_unlimited_pages_and_render_dpi(monkeypatch):
    captured = {}

    def fake_pdf_chunks(pdf_bytes, filename, config, max_pages=None, render_dpi=None):
        captured.update(max_pages=max_pages, render_dpi=render_dpi, filename=filename)
        return [DocumentChunk(id="all-pages", text="完整 PDF 内容", source=filename, page=1)]

    monkeypatch.setattr("manufacturing_rag.api.pdf_bytes_to_chunks", fake_pdf_chunks)
    client = build_client()
    response = client.post(
        "/kb/pdf",
        json={
            "filename": "31-pages.pdf",
            "content_base64": base64.b64encode(b"%PDF-1.4\n%%EOF").decode("ascii"),
            "mode": "replace",
            "max_pages": 0,
            "render_dpi": 150,
        },
    )

    assert response.status_code == 200
    assert captured == {"max_pages": 0, "render_dpi": 150, "filename": "31-pages.pdf"}


def test_kb_chunks_rejects_invalid_mode():
    client = build_client()

    response = client.post("/kb/chunks", json={"mode": "merge", "chunks": []})

    assert response.status_code == 400


def test_image_ocr_ingest_accepts_client_ocr_text():
    client = build_client()
    image_bytes = b"fake-image-bytes"

    response = client.post(
        "/kb/image-ocr",
        json={
            "filename": "page.png",
            "content_base64": base64.b64encode(image_bytes).decode("ascii"),
            "ocr_text": "线性代数资料：矩阵可逆的充要条件包括行列式不等于零。",
            "mode": "replace",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunk_count"] == 1
    assert payload["summary"]["doc_types"] == ["图片OCR"]


def test_image_ocr_ingest_reports_missing_baidu_credentials_without_ocr_text(monkeypatch):
    monkeypatch.delenv("BAIDU_OCR_API_KEY", raising=False)
    monkeypatch.delenv("BAIDU_OCR_SECRET_KEY", raising=False)
    client = build_client()

    response = client.post(
        "/kb/image-ocr",
        json={
            "filename": "page.png",
            "content_base64": base64.b64encode(b"fake-image-bytes").decode("ascii"),
            "mode": "replace",
        },
    )

    assert response.status_code == 503
    assert "百度 OCR" in response.json()["detail"]


def test_vector_ingest_replaces_remote_collection(monkeypatch):
    calls = {}

    class FakeVectorStore:
        def __init__(self, config):
            self.config = config

        def replace_chunks(self, chunks):
            chunk_list = list(chunks)
            calls["chunks"] = chunk_list
            return {"status": "ok", "inserted": len(chunk_list), "collection_name": "fake", "mode": "replace"}

        def upsert_chunks(self, chunks):
            chunk_list = list(chunks)
            calls["chunks"] = chunk_list
            return {"status": "ok", "inserted": len(chunk_list), "collection_name": "fake"}

    monkeypatch.setattr("manufacturing_rag.api.MilvusVectorStore", FakeVectorStore)
    client = build_client(config=replace(DEFAULT_CONFIG, vector_store_provider="zilliz"))

    response = client.post("/vector/ingest", json={"mode": "replace"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "replace"
    assert payload["inserted"] == 1
    assert calls["chunks"][0].id == "api-course-1"


def test_vector_ingest_defaults_to_append_without_dropping_collection(monkeypatch):
    calls = {}

    class FakeVectorStore:
        def __init__(self, config):
            self.config = config

        def upsert_chunks(self, chunks):
            chunk_list = list(chunks)
            calls["chunks"] = chunk_list
            return {"status": "ok", "inserted": len(chunk_list), "collection_name": "fake"}

        def replace_chunks(self, chunks):
            raise AssertionError("append sync must not rebuild the remote collection")

    monkeypatch.setattr("manufacturing_rag.api.MilvusVectorStore", FakeVectorStore)
    client = build_client(config=replace(DEFAULT_CONFIG, vector_store_provider="zilliz"))

    response = client.post("/vector/ingest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "append"
    assert payload["inserted"] == 1
    assert calls["chunks"][0].id == "api-course-1"


def test_vector_error_returns_reason_and_suggestion(monkeypatch):
    class BrokenVectorStore:
        def __init__(self, config):
            self.config = config

        def ensure_collection(self, drop_existing=False):
            raise RuntimeError(
                "MilvusException: Fail connecting to server on example.zilliz.com:443, "
                "illegal connection params or server unavailable"
            )

    monkeypatch.setattr("manufacturing_rag.api.MilvusVectorStore", BrokenVectorStore)
    client = build_client(config=replace(DEFAULT_CONFIG, vector_store_provider="zilliz"))

    response = client.post("/vector/collection", json={"drop_existing": True})

    assert response.status_code == 503
    payload = response.json()
    assert "原因：" in payload["detail"]
    assert payload["error"]["code"] == "VECTOR_CONNECTION_FAILED"
    assert "Zilliz" in payload["error"]["reason"]
    assert payload["error"]["suggestion"]


def test_validation_error_returns_reason_and_suggestion():
    client = build_client()

    response = client.post("/answer", json={})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "INVALID_REQUEST"
    assert "请求参数" in payload["error"]["reason"]
    assert payload["error"]["suggestion"]


def test_unhandled_upload_error_keeps_reason_and_cors_header(monkeypatch):
    client = build_client()

    def fail_upload(*_args, **_kwargs):
        raise ValueError("unexpected PDF parser failure")

    monkeypatch.setattr(client.app.state.kb, "ingest_pdf", fail_upload)
    response = client.post(
        "/kb/pdf",
        headers={"Origin": "http://127.0.0.1:5173"},
        json={"filename": "broken.pdf", "content_base64": base64.b64encode(b"%PDF-1.4").decode("ascii"), "mode": "replace"},
    )

    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    payload = response.json()
    assert "原因：" in payload["detail"]
    assert payload["error"]["technical_detail"] == "unexpected PDF parser failure"
