from __future__ import annotations

from fastapi.testclient import TestClient

from manufacturing_rag.api import create_app
from manufacturing_rag.auth.store import AuthStore


ORIGIN = "http://testserver"
ADMIN_PASSWORD = "A secure passphrase 2026!"
USER_PASSWORD = "Distinct phrase! 2026 account"


def _headers(csrf: str | None = None) -> dict[str, str]:
    headers = {"Origin": ORIGIN, "Content-Type": "application/json"}
    if csrf:
        headers["X-CSRF-Token"] = csrf
    return headers


def _register(client: TestClient, username: str, invite_code: str) -> str:
    response = client.post(
        "/api/auth/register",
        headers=_headers(),
        json={"username": username, "password": USER_PASSWORD, "invite_code": invite_code},
    )
    assert response.status_code == 201
    return response.json()["user"]["id"]


def _login(app, username: str, password: str) -> tuple[TestClient, str, str]:
    client = TestClient(app)
    response = client.post(
        "/api/auth/login",
        headers=_headers(),
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return client, response.json()["user"]["id"], response.json()["csrf_token"]


def test_legacy_data_routes_require_session_and_workspace_permission(monkeypatch):
    monkeypatch.setenv("AUTH_ALLOWED_ORIGINS", ORIGIN)
    monkeypatch.setenv("RAG_CORS_ORIGINS", ORIGIN)
    monkeypatch.setenv("RAG_AUTH_ENFORCEMENT_ENABLED", "true")
    app = create_app()
    store = AuthStore("sqlite://", b"p" * 32)
    app.state.auth_store = store
    try:
        admin = store.create_initial_platform_admin("admin", ADMIN_PASSWORD)
        invite, _status = store.reset_invitation(admin["id"])
        store.set_registration_enabled(admin["id"], True)
        bootstrap_client = TestClient(app)
        _owner_id = _register(bootstrap_client, "workspaceowner", invite)
        reader_id = _register(bootstrap_client, "workspacereader", invite)
        owner, owner_id, owner_csrf = _login(app, "workspaceowner", USER_PASSWORD)
        reader, _reader_id, reader_csrf = _login(app, "workspacereader", USER_PASSWORD)
        anonymous = TestClient(app)

        preflight = anonymous.options(
            "/answer",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-csrf-token",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == ORIGIN

        assert anonymous.get("/kb/summary").status_code == 401
        assert anonymous.post(
            "/answer",
            headers=_headers(),
            json={"question": "test", "scope": "common"},
        ).status_code == 401

        created = owner.post(
            "/api/workspaces",
            headers=_headers(owner_csrf),
            json={"name": "受限测试区"},
        )
        assert created.status_code == 201
        workspace_id = created.json()["id"]

        denied_read = reader.get(f"/kb/summary?workspace_id={workspace_id}")
        assert denied_read.status_code == 403
        denied_answer = reader.post(
            "/answer",
            headers=_headers(reader_csrf),
            json={"question": "测试问题", "scope": "workspace", "workspace_id": workspace_id},
        )
        assert denied_answer.status_code == 403
        denied_write = reader.post(
            "/kb/chunks",
            headers=_headers(reader_csrf),
            json={"chunks": [], "mode": "append", "workspace_id": workspace_id},
        )
        assert denied_write.status_code == 403

        shared = owner.put(
            f"/api/workspaces/{workspace_id}/members/{reader_id}",
            headers=_headers(owner_csrf),
            json={"role": "reader"},
        )
        assert shared.status_code == 200
        allowed_summary = reader.get(f"/kb/summary?workspace_id={workspace_id}")
        assert allowed_summary.status_code == 200
        allowed_answer = reader.post(
            "/answer",
            headers=_headers(reader_csrf),
            json={"question": "测试问题", "scope": "workspace", "workspace_id": workspace_id},
        )
        assert allowed_answer.status_code == 200

        denied_general_write = owner.post(
            "/kb/chunks",
            headers=_headers(owner_csrf),
            json={"chunks": [], "mode": "append", "workspace_id": "campus-default"},
        )
        assert denied_general_write.status_code == 403

        allowed_owner_write = owner.post(
            "/kb/chunks",
            headers=_headers(owner_csrf),
            json={"chunks": [], "mode": "append", "workspace_id": workspace_id},
        )
        assert allowed_owner_write.status_code == 200
    finally:
        store.engine.dispose()
