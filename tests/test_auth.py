from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from manufacturing_rag.auth.store import AuthStore, LastPlatformAdminError
from manufacturing_rag.api import create_app


ORIGIN = "http://testserver"
ADMIN_PASSWORD = "A secure passphrase 2026!"
USER_PASSWORD = "Distinct phrase! 2026 account"


@pytest.fixture
def auth_client(monkeypatch):
    monkeypatch.setenv("AUTH_ALLOWED_ORIGINS", ORIGIN)
    monkeypatch.setenv("RAG_CORS_ORIGINS", ORIGIN)
    app = create_app()
    store = AuthStore("sqlite://", b"t" * 32)
    app.state.auth_store = store
    with TestClient(app) as client:
        yield client, store
    store.engine.dispose()


def browser_headers(csrf_token: str | None = None) -> dict[str, str]:
    headers = {"Origin": ORIGIN, "Content-Type": "application/json"}
    if csrf_token:
        headers["X-CSRF-Token"] = csrf_token
    return headers


def bootstrap_admin(store: AuthStore) -> tuple[dict, str]:
    admin = store.create_initial_platform_admin("admin", ADMIN_PASSWORD)
    code, _ = store.reset_invitation(admin["id"])
    store.set_registration_enabled(admin["id"], True)
    return admin, code


def test_register_login_session_and_password_change(auth_client):
    client, store = auth_client
    admin, invite = bootstrap_admin(store)

    registration = client.post(
        "/api/auth/register",
        headers=browser_headers(),
        json={
            "username": "Student_01",
            "display_name": "学生甲",
            "password": USER_PASSWORD,
            "invite_code": invite.lower(),
        },
    )
    assert registration.status_code == 201
    assert registration.json()["user"]["username"] == "student_01"
    assert registration.json()["user"]["platform_role"] == "user"
    cookie = registration.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert "password_hash" not in registration.json()["user"]

    csrf = registration.json()["csrf_token"]
    current = client.get("/api/auth/session")
    assert current.status_code == 200
    assert current.json()["authenticated"] is True
    assert current.json()["csrf_token"] == csrf

    missing_csrf = client.post(
        "/api/auth/change-password",
        headers=browser_headers(),
        json={"current_password": USER_PASSWORD, "new_password": "New strong phrase! 2026 account"},
    )
    assert missing_csrf.status_code == 403

    changed = client.post(
        "/api/auth/change-password",
        headers=browser_headers(csrf),
        json={"current_password": USER_PASSWORD, "new_password": "New strong phrase! 2026 account"},
    )
    assert changed.status_code == 200
    assert client.get("/api/auth/session").json()["authenticated"] is False

    login = client.post(
        "/api/auth/login",
        headers=browser_headers(),
        json={"username": "STUDENT_01", "password": "New strong phrase! 2026 account"},
    )
    assert login.status_code == 200
    assert login.json()["user"]["id"] == registration.json()["user"]["id"]
    assert store.current_principal(client.cookies.get("kujie_session")) is not None


def test_platform_user_controls_and_public_registration_status(auth_client):
    client, store = auth_client
    admin, invite = bootstrap_admin(store)
    assert client.get("/api/auth/registration").json() == {
        "configured": True,
        "registration_enabled": True,
    }
    registered = client.post(
        "/api/auth/register",
        headers=browser_headers(),
        json={"username": "managee", "password": USER_PASSWORD, "invite_code": invite},
    )
    assert registered.status_code == 201
    user_id = registered.json()["user"]["id"]

    admin_client = TestClient(client.app)
    admin_login = admin_client.post(
        "/api/auth/login",
        headers=browser_headers(),
        json={"username": "admin", "password": ADMIN_PASSWORD},
    )
    assert admin_login.status_code == 200
    csrf = admin_login.json()["csrf_token"]

    listed = admin_client.get("/api/auth/admin/users?q=managee")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["users"]] == [user_id]
    assert "password_hash" not in listed.json()["users"][0]
    assert client.get("/api/auth/admin/users").status_code == 403

    promoted = admin_client.patch(
        f"/api/auth/admin/users/{user_id}/platform-role",
        headers=browser_headers(csrf),
        json={"platform_role": "platform_admin"},
    )
    assert promoted.status_code == 200
    assert client.get("/api/auth/session").json()["authenticated"] is False
    disabled = admin_client.patch(
        f"/api/auth/admin/users/{user_id}/status",
        headers=browser_headers(csrf),
        json={"status": "suspended"},
    )
    assert disabled.status_code == 200
    assert client.get("/api/auth/session").json()["authenticated"] is False

    with pytest.raises(LastPlatformAdminError, match="至少保留一名启用的平台管理员"):
        store.set_platform_role(admin["id"], admin["id"], "user")


def test_invite_reset_is_single_display_and_admin_only(auth_client):
    client, store = auth_client
    admin, old_code = bootstrap_admin(store)
    admin_client = TestClient(client.app)
    login = admin_client.post(
        "/api/auth/login",
        headers=browser_headers(),
        json={"username": "admin", "password": ADMIN_PASSWORD},
    )
    assert login.status_code == 200

    status = admin_client.get("/api/auth/admin/invitation")
    assert status.status_code == 200
    assert status.json()["configured"] is True
    assert "invite_code" not in status.json()

    reset = admin_client.post(
        "/api/auth/admin/invitation/reset",
        headers=browser_headers(login.json()["csrf_token"]),
        json={"confirm": True},
    )
    assert reset.status_code == 200
    new_code = reset.json()["invite_code"]
    assert len(new_code) == 6 and new_code != old_code

    rejected = admin_client.post(
        "/api/auth/register",
        headers=browser_headers(),
        json={
            "username": "student02",
            "password": USER_PASSWORD,
            "invite_code": old_code,
        },
    )
    assert rejected.status_code == 400

    accepted = admin_client.post(
        "/api/auth/register",
        headers=browser_headers(),
        json={
            "username": "student02",
            "password": USER_PASSWORD,
            "invite_code": new_code,
        },
    )
    assert accepted.status_code == 201

    anonymous = TestClient(client.app)
    denied = anonymous.get("/api/auth/admin/invitation")
    assert denied.status_code == 401


def test_registration_rejects_role_injection_and_untrusted_origin(auth_client):
    client, store = auth_client
    _admin, invite = bootstrap_admin(store)
    injected = client.post(
        "/api/auth/register",
        headers=browser_headers(),
        json={
            "username": "student03",
            "password": USER_PASSWORD,
            "invite_code": invite,
            "platform_role": "platform_admin",
            "department_id": "department-from-client",
        },
    )
    assert injected.status_code == 422

    untrusted = client.post(
        "/api/auth/register",
        headers={"Origin": "https://attacker.example", "Content-Type": "application/json"},
        json={
            "username": "student03",
            "password": USER_PASSWORD,
            "invite_code": invite,
        },
    )
    assert untrusted.status_code == 403


def test_login_errors_are_generic_and_rate_limited(auth_client):
    client, store = auth_client
    errors = [
        client.post(
            "/api/auth/login",
            headers=browser_headers(),
            json={"username": "unknown_user", "password": USER_PASSWORD},
        )
        for _ in range(5)
    ]
    assert all(response.status_code == 401 for response in errors)
    assert all(response.json()["detail"] == errors[0].json()["detail"] for response in errors)
    limited = client.post(
        "/api/auth/login",
        headers=browser_headers(),
        json={"username": "unknown_user", "password": USER_PASSWORD},
    )
    assert limited.status_code == 429

    store.create_initial_platform_admin("admin", ADMIN_PASSWORD)
    admin_client = TestClient(client.app)
    admin_login = admin_client.post(
        "/api/auth/login",
        headers=browser_headers(),
        json={"username": "admin", "password": ADMIN_PASSWORD},
    )
    csrf = admin_login.json()["csrf_token"]
    cleared = admin_client.post(
        "/api/auth/admin/login-limits/clear",
        headers=browser_headers(csrf),
        json={"username": "unknown_user", "confirm": True},
    )
    assert cleared.status_code == 200
    assert cleared.json()["removed_bucket_count"] == 2
    after_clear = client.post(
        "/api/auth/login",
        headers=browser_headers(),
        json={"username": "unknown_user", "password": USER_PASSWORD},
    )
    assert after_clear.status_code == 401


def test_logout_all_revokes_sessions_across_devices(auth_client):
    _primary_client, store = auth_client
    admin = store.create_initial_platform_admin("admin", ADMIN_PASSWORD)
    primary = TestClient(_primary_client.app)
    second_device = TestClient(_primary_client.app)
    for device in (primary, second_device):
        response = device.post(
            "/api/auth/login",
            headers=browser_headers(),
            json={"username": "admin", "password": ADMIN_PASSWORD},
        )
        assert response.status_code == 200
    csrf = primary.get("/api/auth/session").json()["csrf_token"]
    logout = primary.post("/api/auth/logout-all", headers=browser_headers(csrf))
    assert logout.status_code == 200
    assert primary.get("/api/auth/session").json()["authenticated"] is False
    assert second_device.get("/api/auth/session").json()["authenticated"] is False
    assert admin["platform_role"] == "platform_admin"


def test_browser_preflight_allows_credentials_only_for_configured_origin(auth_client):
    client, _store = auth_client
    response = client.options(
        "/api/auth/login",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-csrf-token",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_password_normalization_and_rules():
    from manufacturing_rag.auth.security import normalize_password, validate_new_password

    assert normalize_password("e\u0301" * 15) == "é" * 15
    with pytest.raises(ValueError):
        normalize_password("short")
    with pytest.raises(ValueError):
        normalize_password("valid password 123\n")
    with pytest.raises(ValueError):
        validate_new_password("Student_01 secure phrase 2026!", "student_01")
    with pytest.raises(ValueError):
        validate_new_password("a strong password 2026!", "admin")
