from dataclasses import replace

from manufacturing_rag.api import DEFAULT_CONFIG, create_app, explain_api_error
from manufacturing_rag.auth.store import AuthStore
from fastapi.testclient import TestClient


def test_repeated_delete_reports_missing_conversation_without_api_configuration_hint(monkeypatch):
    monkeypatch.setenv("AUTH_ALLOWED_ORIGINS", "http://testserver")
    monkeypatch.setenv("RAG_CORS_ORIGINS", "http://testserver")
    app = create_app(config=replace(DEFAULT_CONFIG, vector_store_provider="memory"))
    store = AuthStore("sqlite://", b"c" * 32)
    app.state.auth_store = store
    try:
        store.create_initial_platform_admin("admin", "Admin!2026abc")
        with TestClient(app) as client:
            headers = {"Origin": "http://testserver", "Content-Type": "application/json"}
            login = client.post("/api/auth/login", headers=headers, json={"username": "admin", "password": "Admin!2026abc"})
            assert login.status_code == 200
            headers["X-CSRF-Token"] = login.json()["csrf_token"]
            created = client.post("/api/conversations", headers=headers, json={"scope_type": "common"})
            assert created.status_code == 201
            url = "/api/conversations/" + created.json()["conversation"]["id"]
            assert client.delete(url, headers=headers).status_code == 200
            for response in (client.delete(url, headers=headers), client.get(url)):
                assert response.status_code == 404
                assert response.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"
                assert "API" not in response.json()["detail"]
    finally:
        store.engine.dispose()


def test_resource_not_found_retains_its_reason():
    error = explain_api_error("原始文档不存在，请重新上传后预览", 404)
    assert error["code"] == "RESOURCE_NOT_FOUND"
    assert "原始文档不存在" in error["reason"]
    assert explain_api_error("Not Found", 404)["code"] == "API_NOT_FOUND"
