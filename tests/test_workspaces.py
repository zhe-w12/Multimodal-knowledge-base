from __future__ import annotations

from fastapi.testclient import TestClient
from fastapi import HTTPException
from sqlalchemy import select

from manufacturing_rag.auth.store import AuthStore
from manufacturing_rag.auth.models import AuditLog, ConversationMessage
from manufacturing_rag.auth.conversation_store import ConversationStore
from manufacturing_rag.api import create_app
from manufacturing_rag.schemas import DEFAULT_WORKSPACE_ID, DocumentChunk


ORIGIN = "http://testserver"
ADMIN_PASSWORD = "Admin!2026abc"
USER_PASSWORD = "User!2026abc"


def _headers(csrf_token: str | None = None) -> dict[str, str]:
    headers = {"Origin": ORIGIN, "Content-Type": "application/json"}
    if csrf_token:
        headers["X-CSRF-Token"] = csrf_token
    return headers


def _create_account(client: TestClient, username: str) -> tuple[str, str]:
    response = client.post(
        "/api/auth/register",
        headers=_headers(),
        json={"username": username, "password": USER_PASSWORD},
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
    app = create_app(initial_chunks=[])
    store = AuthStore("sqlite://", b"w" * 32)
    app.state.auth_store = store
    admin = store.create_initial_platform_admin("admin", ADMIN_PASSWORD)
    bootstrap_client = TestClient(app)
    user_a_id, user_a_csrf = _create_account(bootstrap_client, "workspace_owner")
    user_b_id, user_b_csrf = _create_account(bootstrap_client, "workspace_reader")
    user_c_id, _user_c_csrf = _create_account(bootstrap_client, "workspace_other")
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


def test_member_role_preview_and_stale_workspace_version(monkeypatch):
    (
        _app,
        store,
        owner,
        _owner_id,
        owner_csrf,
        reader,
        reader_id,
        _reader_csrf,
        _other_id,
        _admin_client,
        _admin_csrf,
    ) = _setup(monkeypatch)
    try:
        workspace = owner.post(
            "/api/workspaces",
            headers=_headers(owner_csrf),
            json={"name": "权限预览测试区"},
        ).json()
        workspace_url = f"/api/workspaces/{workspace['id']}"
        expected_version = workspace["version"]

        preview = owner.post(
            f"{workspace_url}/members/{reader_id}/preview",
            headers=_headers(owner_csrf),
            json={"role": "reader", "expected_workspace_version": expected_version},
        )
        assert preview.status_code == 200
        assert preview.json()["before"]["can_read"] is False
        assert preview.json()["after"]["can_read"] is True

        saved = owner.put(
            f"{workspace_url}/members/{reader_id}",
            headers=_headers(owner_csrf),
            json={"role": "reader", "expected_workspace_version": expected_version},
        )
        assert saved.status_code == 200
        member = next(
            item for item in owner.get(f"{workspace_url}/members").json()["members"]
            if item["user_id"] == reader_id
        )
        assert member["permission_source"] == "个人设置"
        assert member["can_read"] is True

        stale = owner.put(
            f"{workspace_url}/members/{reader_id}",
            headers=_headers(owner_csrf),
            json={"role": "blocked", "expected_workspace_version": expected_version},
        )
        assert stale.status_code == 409
        assert reader.get(f"{workspace_url}/members").status_code == 403
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


def test_workspace_creator_can_permanently_delete_after_transfer(monkeypatch, tmp_path):
    (
        app,
        store,
        creator,
        creator_id,
        creator_csrf,
        reader,
        reader_id,
        reader_csrf,
        _other_id,
        admin_client,
        _admin_csrf,
    ) = _setup(monkeypatch)
    try:
        monkeypatch.setattr("manufacturing_rag.api.DOCUMENT_FILES_DIR", tmp_path)
        workspace = creator.post(
            "/api/workspaces",
            headers=_headers(creator_csrf),
            json={"name": "待删除资料区"},
        ).json()
        workspace_id = workspace["id"]

        denied = reader.request(
            "DELETE",
            f"/api/workspaces/{workspace_id}",
            headers=_headers(reader_csrf),
            json={"confirm": True, "workspace_name": workspace["name"]},
        )
        assert denied.status_code == 403

        missing_csrf = creator.request(
            "DELETE",
            f"/api/workspaces/{workspace_id}",
            headers={"Origin": ORIGIN, "Content-Type": "application/json"},
            json={"confirm": True, "workspace_name": workspace["name"]},
        )
        assert missing_csrf.status_code == 403

        transferred = creator.post(
            f"/api/workspaces/{workspace_id}/transfer",
            headers=_headers(creator_csrf),
            json={"new_owner_user_id": reader_id},
        )
        assert transferred.status_code == 200

        wrong_name = creator.request(
            "DELETE",
            f"/api/workspaces/{workspace_id}",
            headers=_headers(creator_csrf),
            json={"confirm": True, "workspace_name": "错误名称"},
        )
        assert wrong_name.status_code == 422
        assert creator.get(f"/api/workspaces/{workspace_id}").status_code == 200

        blocked = reader.put(
            f"/api/workspaces/{workspace_id}/members/{creator_id}",
            headers=_headers(reader_csrf),
            json={"role": "blocked"},
        )
        assert blocked.status_code == 200
        creator_access = next(
            item for item in creator.get("/api/workspaces").json()["workspaces"]
            if item["id"] == workspace_id
        )
        assert creator_access["can_delete"] is True
        assert creator_access["can_manage"] is False

        app.state.kb.update_chunks(
            [
                {"id": "private-chunk", "text": "私有资料", "source": "private.pdf", "metadata": {"workspace_id": workspace_id}},
            ],
            workspace_id=workspace_id,
        )
        app.state.kb.update_chunks(
            [{"id": "general-chunk", "text": "通用资料", "source": "general.pdf"}],
            workspace_id=DEFAULT_WORKSPACE_ID,
        )
        assert app.state.kb.workspace_chunk_ids(workspace_id)
        from manufacturing_rag.api import stored_document_metadata_path, stored_document_path

        original = stored_document_path(workspace_id, "private.pdf")
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(b"private source")
        stored_document_metadata_path(workspace_id, "private.pdf").write_text("{}", encoding="utf-8")

        conversation = ConversationStore(store).create(creator_id, "workspace", workspace_id)
        ConversationStore(store).begin_turn(
            creator_id,
            conversation["id"],
            "workspace",
            workspace_id,
            [workspace_id],
            "私有问题",
        )
        ConversationStore(store).append_assistant(creator_id, conversation["id"], "私有回答", [])
        all_scope_with_workspace_source = ConversationStore(store).create(creator_id, "all")
        ConversationStore(store).append_assistant(
            creator_id,
            all_scope_with_workspace_source["id"],
            "引用授权工作区资料的回答",
            [{"metadata": {"workspace_id": workspace_id}}],
        )
        all_scope_with_general_source = ConversationStore(store).create(creator_id, "all")
        ConversationStore(store).append_assistant(
            creator_id,
            all_scope_with_general_source["id"],
            "引用通用资料的回答",
            [{"metadata": {"workspace_id": DEFAULT_WORKSPACE_ID}}],
        )

        original_cleanup = app.state.delete_workspace_data

        def fail_cleanup(_workspace_id):
            raise HTTPException(status_code=503, detail="模拟索引删除失败")

        app.state.delete_workspace_data = fail_cleanup
        failed = creator.request(
            "DELETE",
            f"/api/workspaces/{workspace_id}",
            headers=_headers(creator_csrf),
            json={"confirm": True, "workspace_name": workspace["name"]},
        )
        assert failed.status_code == 503
        archived_workspace = next(
            item for item in creator.get("/api/workspaces").json()["workspaces"]
            if item["id"] == workspace_id
        )
        assert archived_workspace["status"] == "archived"
        assert archived_workspace["deletion_in_progress"] is True
        assert original.exists()
        blocked_restore = reader.post(
            f"/api/workspaces/{workspace_id}/restore",
            headers=_headers(reader_csrf),
            json={"confirm": True},
        )
        assert blocked_restore.status_code == 409

        app.state.delete_workspace_data = original_cleanup
        deleted = creator.request(
            "DELETE",
            f"/api/workspaces/{workspace_id}",
            headers=_headers(creator_csrf),
            json={"confirm": True, "workspace_name": workspace["name"]},
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted_conversation_count"] == 2
        assert deleted.json()["removed_chunk_count"] > 0
        assert not app.state.kb.workspace_chunk_ids(workspace_id), deleted.json()
        assert creator.get(f"/api/workspaces/{workspace_id}").status_code == 404
        assert all(item["id"] != workspace_id for item in creator.get("/api/workspaces").json()["workspaces"])
        remaining_chunks = app.state.kb.chunks_snapshot()
        assert [(chunk.source, chunk.metadata.get("workspace_id")) for chunk in remaining_chunks] == [
            ("general.pdf", DEFAULT_WORKSPACE_ID)
        ]
        assert not original.parent.exists()
        remaining_conversation_ids = {
            item["id"] for item in creator.get("/api/conversations").json()["conversations"]
        }
        assert conversation["id"] not in remaining_conversation_ids
        assert all_scope_with_workspace_source["id"] not in remaining_conversation_ids
        assert all_scope_with_general_source["id"] in remaining_conversation_ids
        with store.Session() as db:
            assert db.scalar(select(AuditLog.id).where(AuditLog.action == "workspace_deleted", AuditLog.target_id == workspace_id))
            assert db.scalar(select(ConversationMessage.id).where(ConversationMessage.conversation_id == conversation["id"])) is None
            assert db.scalar(select(ConversationMessage.id).where(ConversationMessage.conversation_id == all_scope_with_workspace_source["id"])) is None
            assert db.scalar(select(ConversationMessage.id).where(ConversationMessage.conversation_id == all_scope_with_general_source["id"])) is not None
        assert reader.get("/api/workspaces").status_code == 200
    finally:
        store.engine.dispose()


def test_general_workspace_cannot_be_permanently_deleted(monkeypatch):
    (
        _app,
        store,
        _creator,
        _creator_id,
        _creator_csrf,
        _reader,
        _reader_id,
        _reader_csrf,
        _other_id,
        admin_client,
        admin_csrf,
    ) = _setup(monkeypatch)
    try:
        general = next(
            item for item in admin_client.get("/api/workspaces").json()["workspaces"]
            if item["workspace_type"] == "general"
        )
        response = admin_client.request(
            "DELETE",
            f"/api/workspaces/{general['id']}",
            headers=_headers(admin_csrf),
            json={"confirm": True, "workspace_name": general["name"]},
        )
        assert response.status_code == 403
        assert admin_client.get(f"/api/workspaces/{general['id']}").status_code == 200
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
