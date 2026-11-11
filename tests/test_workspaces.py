from __future__ import annotations

from fastapi.testclient import TestClient

from manufacturing_rag.auth.store import AuthStore
from manufacturing_rag.api import create_app


ORIGIN = "http://testserver"
ADMIN_PASSWORD = "A secure passphrase 2026!"
USER_PASSWORD = "Distinct phrase! 2026 account"


def _headers(csrf_token: str | None = None) -> dict[str, str]:
    headers = {"Origin": ORIGIN, "Content-Type": "application/json"}
    if csrf_token:
        headers["X-CSRF-Token"] = csrf_token
    return headers


def _create_account(client: TestClient, username: str, invite_code: str) -> tuple[str, str]:
    response = client.post(
        "/api/auth/register",
        headers=_headers(),
        json={"username": username, "password": USER_PASSWORD, "invite_code": invite_code},
    )
    assert response.status_code == 201
    body = response.json()
    return body["user"]["id"], body["csrf_token"]


def _login(app, username: str, password: str) -> tuple[TestClient, str, str]:
    client = TestClient(app)
    response = client.post(
        "/api/auth/login",
        headers=_headers(),
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return client, response.json()["user"]["id"], response.json()["csrf_token"]


def _setup(monkeypatch):
    monkeypatch.setenv("AUTH_ALLOWED_ORIGINS", ORIGIN)
    monkeypatch.setenv("RAG_CORS_ORIGINS", ORIGIN)
    app = create_app()
    store = AuthStore("sqlite://", b"w" * 32)
    app.state.auth_store = store
    admin = store.create_initial_platform_admin("admin", ADMIN_PASSWORD)
    invite, _status = store.reset_invitation(admin["id"])
    store.set_registration_enabled(admin["id"], True)
    bootstrap_client = TestClient(app)
    user_a_id, user_a_csrf = _create_account(bootstrap_client, "workspace_owner", invite)
    user_b_id, user_b_csrf = _create_account(bootstrap_client, "workspace_reader", invite)
    user_c_id, _user_c_csrf = _create_account(bootstrap_client, "workspace_other", invite)
    user_a, _, user_a_csrf = _login(app, "workspace_owner", USER_PASSWORD)
    user_b, _, user_b_csrf = _login(app, "workspace_reader", USER_PASSWORD)
    admin_client, _, admin_csrf = _login(app, "admin", ADMIN_PASSWORD)
    return app, store, user_a, user_a_id, user_a_csrf, user_b, user_b_id, user_b_csrf, user_c_id, admin_client, admin_csrf


def test_audit_logs_are_scoped_to_platform_admin_and_workspace_manager(monkeypatch):
    (
        _app,
        store,
        user_a,
        _user_a_id,
        user_a_csrf,
        user_b,
        _user_b_id,
        user_b_csrf,
        _user_c_id,
        admin_client,
        _admin_csrf,
    ) = _setup(monkeypatch)
    try:
        workspace_a = user_a.post(
            "/api/workspaces",
            headers=_headers(user_a_csrf),
            json={"name": "审计范围甲", "description": ""},
        ).json()
        workspace_b = user_b.post(
            "/api/workspaces",
            headers=_headers(user_b_csrf),
            json={"name": "审计范围乙", "description": ""},
        ).json()

        owner_audit = user_a.get(f"/api/workspaces/{workspace_a['id']}/audit")
        assert owner_audit.status_code == 200
        owner_entries = owner_audit.json()["entries"]
        assert any(entry["action"] == "workspace_created" for entry in owner_entries)
        assert all(entry["target_id"] == workspace_a["id"] for entry in owner_entries)

        cross_workspace = user_a.get(f"/api/workspaces/{workspace_b['id']}/audit")
        assert cross_workspace.status_code == 403
        user_global_audit = user_b.get("/api/auth/admin/audit")
        assert user_global_audit.status_code == 403

        platform_audit = admin_client.get("/api/auth/admin/audit")
        assert platform_audit.status_code == 200
        platform_entries = platform_audit.json()["entries"]
        assert any(entry["target_id"] == workspace_a["id"] for entry in platform_entries)
        assert any(entry["target_id"] == workspace_b["id"] for entry in platform_entries)
    finally:
        store.engine.dispose()


def test_any_active_user_can_create_an_authorized_workspace(monkeypatch):
    (
        _app,
        store,
        user_a,
        user_a_id,
        user_a_csrf,
        user_b,
        _user_b_id,
        user_b_csrf,
        _user_c_id,
        _admin_client,
        _admin_csrf,
    ) = _setup(monkeypatch)
    try:
        general = user_a.get("/api/workspaces")
        assert general.status_code == 200
        assert [item["id"] for item in general.json()["workspaces"]] == ["campus-default"]

        created = user_a.post(
            "/api/workspaces",
            headers=_headers(user_a_csrf),
            json={"name": "  研发一组  ", "description": "授权资料"},
        )
        assert created.status_code == 201
        workspace = created.json()
        assert workspace["workspace_type"] == "authorized"
        assert workspace["owner_user_id"] == user_a_id
        assert workspace["role"] == "owner"
        assert workspace["can_manage"] is True
        assert user_a.get("/api/auth/session").json()["user"]["platform_role"] == "user"

        denied = user_b.get(f"/api/workspaces/{workspace['id']}")
        assert denied.status_code == 403
        assert all(item["id"] != workspace["id"] for item in user_b.get("/api/workspaces").json()["workspaces"])

        granted = user_a.put(
            f"/api/workspaces/{workspace['id']}/members/{_user_b_id}",
            headers=_headers(user_a_csrf),
            json={"role": "reader"},
        )
        assert granted.status_code == 200
        visible = user_b.get("/api/workspaces").json()["workspaces"]
        shared = next(item for item in visible if item["id"] == workspace["id"])
        assert shared["role"] == "reader" and shared["can_read"] is True and shared["can_manage"] is False

        blocked = user_a.put(
            f"/api/workspaces/{workspace['id']}/members/{_user_b_id}",
            headers=_headers(user_a_csrf),
            json={"role": "blocked"},
        )
        assert blocked.status_code == 200
        assert all(item["id"] != workspace["id"] for item in user_b.get("/api/workspaces").json()["workspaces"])
    finally:
        store.engine.dispose()


def test_workspace_manager_can_find_only_active_user_by_exact_username(monkeypatch):
    (
        _app,
        store,
        owner,
        _owner_id,
        _owner_csrf,
        reader,
        reader_id,
        _reader_csrf,
        _other_id,
        _admin_client,
        _admin_csrf,
    ) = _setup(monkeypatch)
    try:
        created = owner.post(
            "/api/workspaces",
            headers=_headers(_owner_csrf),
            json={"name": "成员查询测试区"},
        )
        workspace_id = created.json()["id"]

        found = owner.get(
            f"/api/workspaces/{workspace_id}/users/search",
            params={"username": "WORKSPACE_READER"},
        )
        assert found.status_code == 200
        assert found.json()["user"] == {
            "user_id": reader_id,
            "username": "workspace_reader",
            "display_name": "workspace_reader",
            "department_id": None,
        }
        denied = reader.get(
            f"/api/workspaces/{workspace_id}/users/search",
            params={"username": "workspace_other"},
        )
        assert denied.status_code == 403
    finally:
        store.engine.dispose()


def test_department_grants_are_read_only_and_direct_block_overrides_them(monkeypatch):
    (
        _app,
        store,
        user_a,
        user_a_id,
        user_a_csrf,
        user_b,
        user_b_id,
        user_b_csrf,
        _user_c_id,
        admin_client,
        admin_csrf,
    ) = _setup(monkeypatch)
    try:
        workspace = user_a.post(
            "/api/workspaces",
            headers=_headers(user_a_csrf),
            json={"name": "部门资料区"},
        ).json()
        department = admin_client.post(
            "/api/admin/departments",
            headers=_headers(admin_csrf),
            json={"name": "研发部"},
        )
        assert department.status_code == 201
        department_id = department.json()["id"]
        assigned = admin_client.patch(
            f"/api/admin/users/{user_b_id}/department",
            headers=_headers(admin_csrf),
            json={"department_id": department_id},
        )
        assert assigned.status_code == 200

        grant = user_a.put(
            f"/api/workspaces/{workspace['id']}/departments/{department_id}",
            headers=_headers(user_a_csrf),
            json={"enabled": True},
        )
        assert grant.status_code == 200
        inherited = user_b.get("/api/workspaces").json()["workspaces"]
        inherited_workspace = next(item for item in inherited if item["id"] == workspace["id"])
        assert inherited_workspace["role"] == "department_reader"
        assert inherited_workspace["can_read"] is True
        assert inherited_workspace["can_manage"] is False

        owner_promote = user_a.put(
            f"/api/workspaces/{workspace['id']}/members/{user_b_id}",
            headers=_headers(user_a_csrf),
            json={"role": "admin"},
        )
        assert owner_promote.status_code == 200
        # A workspace admin can manage read grants, but cannot elevate another user to admin.
        denied_admin = user_b.put(
            f"/api/workspaces/{workspace['id']}/members/{user_a_id}",
            headers=_headers(user_b_csrf),
            json={"role": "admin"},
        )
        assert denied_admin.status_code == 403

        blocked = user_a.put(
            f"/api/workspaces/{workspace['id']}/members/{user_b_id}",
            headers=_headers(user_a_csrf),
            json={"role": "blocked"},
        )
        assert blocked.status_code == 200
        assert all(item["id"] != workspace["id"] for item in user_b.get("/api/workspaces").json()["workspaces"])
    finally:
        store.engine.dispose()


def test_workspace_owner_controls_administrators_transfer_and_archive(monkeypatch):
    (
        app,
        store,
        user_a,
        user_a_id,
        user_a_csrf,
        user_b,
        user_b_id,
        user_b_csrf,
        user_c_id,
        _admin_client,
        _admin_csrf,
    ) = _setup(monkeypatch)
    try:
        workspace = user_a.post(
            "/api/workspaces",
            headers=_headers(user_a_csrf),
            json={"name": "负责人变更区"},
        ).json()
        promote = user_a.put(
            f"/api/workspaces/{workspace['id']}/members/{user_b_id}",
            headers=_headers(user_a_csrf),
            json={"role": "admin"},
        )
        assert promote.status_code == 200
        denied_promotion = user_b.put(
            f"/api/workspaces/{workspace['id']}/members/{user_c_id}",
            headers=_headers(user_b_csrf),
            json={"role": "admin"},
        )
        assert denied_promotion.status_code == 403

        transferred = user_a.post(
            f"/api/workspaces/{workspace['id']}/transfer",
            headers=_headers(user_a_csrf),
            json={"new_owner_user_id": user_b_id},
        )
        assert transferred.status_code == 200
        assert transferred.json()["owner_user_id"] == user_b_id

        archived = user_b.post(
            f"/api/workspaces/{workspace['id']}/archive",
            headers=_headers(user_b_csrf),
            json={"confirm": True},
        )
        assert archived.status_code == 200
        assert archived.json()["status"] == "archived"
        archived_for_old_owner = next(
            item for item in user_a.get("/api/workspaces").json()["workspaces"] if item["id"] == workspace["id"]
        )
        assert archived_for_old_owner["can_read"] is False
        assert archived_for_old_owner["can_manage"] is True

        restored = user_b.post(
            f"/api/workspaces/{workspace['id']}/restore",
            headers=_headers(user_b_csrf),
            json={"confirm": True},
        )
        assert restored.status_code == 200
        assert restored.json()["status"] == "active"
        assert restored.json()["owner_user_id"] == user_b_id
        assert user_a.get(f"/api/workspaces/{workspace['id']}").status_code == 200
        assert user_a_id != user_b_id
    finally:
        store.engine.dispose()


def test_workspace_payload_cannot_choose_general_type_or_forge_owner(monkeypatch):
    (
        _app,
        store,
        user_a,
        _user_a_id,
        user_a_csrf,
        _user_b,
        _user_b_id,
        _user_b_csrf,
        _user_c_id,
        _admin_client,
        _admin_csrf,
    ) = _setup(monkeypatch)
    try:
        response = user_a.post(
            "/api/workspaces",
            headers=_headers(user_a_csrf),
            json={"name": "伪造通用区", "workspace_type": "general", "owner_user_id": "another-user"},
        )
        assert response.status_code == 422
        assert [item["id"] for item in user_a.get("/api/workspaces").json()["workspaces"]] == ["campus-default"]
    finally:
        store.engine.dispose()
