from __future__ import annotations

import hmac
import os
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from manufacturing_rag.auth.store import (
    AccountAlreadyExists,
    AccountUnavailable,
    AuthServiceUnavailable,
    AuthStore,
    CurrentPasswordIncorrect,
    InvalidCredentials,
    LastPlatformAdminError,
    LoginTemporarilyLimited,
    NewPasswordUnchanged,
    PlatformAdminRequired,
    Principal,
    RegistrationClosed,
)


router = APIRouter(tags=["身份认证"])
SESSION_COOKIE_NAME = "kujie_session"
_store_creation_lock = threading.Lock()


class StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisterPayload(StrictPayload):
    username: str = Field(min_length=1, max_length=128)
    display_name: str | None = Field(default=None, max_length=120)
    password: str = Field(min_length=1, max_length=512)


class LoginPayload(StrictPayload):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class ChangePasswordPayload(StrictPayload):
    current_password: str = Field(min_length=1, max_length=512)
    new_password: str = Field(min_length=1, max_length=512)


class RegistrationTogglePayload(StrictPayload):
    enabled: bool


class ClearLoginLimitPayload(StrictPayload):
    username: str = Field(min_length=1, max_length=128)
    confirm: bool


class PlatformRolePayload(StrictPayload):
    platform_role: str = Field(pattern="^(user|platform_admin)$")


class UserStatusPayload(StrictPayload):
    status: str = Field(pattern="^(active|suspended)$")


def _auth_store(request: Request) -> AuthStore:
    store = getattr(request.app.state, "auth_store", None)
    if store is not None:
        return store
    with _store_creation_lock:
        store = getattr(request.app.state, "auth_store", None)
        if store is None:
            try:
                store = AuthStore.from_environment()
            except Exception as exc:  # noqa: BLE001 - do not expose database URLs or secret configuration.
                raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc
            request.app.state.auth_store = store
    return store


def _allowed_origins() -> set[str]:
    configured = os.getenv("AUTH_ALLOWED_ORIGINS", "").strip()
    if not configured:
        configured = os.getenv("RAG_CORS_ORIGINS", "").strip()
    return {origin.strip() for origin in configured.split(",") if origin.strip() and origin.strip() != "*"}


def _require_browser_request(request: Request, *, require_json: bool = True) -> None:
    origin = request.headers.get("origin", "")
    if not origin or origin not in _allowed_origins():
        raise HTTPException(status_code=403, detail="请求来源不允许")
    if require_json:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise HTTPException(status_code=415, detail="请求格式无效")


def _request_ip(request: Request) -> str:
    return request.client.host if request.client and request.client.host else "unknown"


def _principal(request: Request, store: AuthStore, *, required: bool = True) -> Principal | None:
    raw_token = request.cookies.get(SESSION_COOKIE_NAME)
    try:
        principal = store.current_principal(raw_token)
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc
    if required and principal is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return principal


def _require_csrf(request: Request, principal: Principal) -> None:
    received = request.headers.get("x-csrf-token", "")
    if not received or not hmac.compare_digest(received, principal.csrf_token):
        raise HTTPException(status_code=403, detail="安全校验失败，请刷新页面后重试")


def _require_platform_admin(principal: Principal) -> None:
    if principal.user.get("platform_role") != "platform_admin":
        raise HTTPException(status_code=403, detail="需要平台管理员权限")


def _cookie_secure(request: Request) -> bool:
    configured = os.getenv("AUTH_COOKIE_SECURE")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    forwarded_protocol = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    return request.url.scheme == "https" or forwarded_protocol == "https"


def _set_session_cookie(response: Response, request: Request, token: str) -> None:
    try:
        max_age = max(1, int(os.getenv("AUTH_SESSION_ABSOLUTE_HOURS", "12"))) * 60 * 60
    except ValueError:
        max_age = 12 * 60 * 60
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=max_age,
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        path="/",
    )


def _clear_session_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=_cookie_secure(request),
        httponly=True,
        samesite="lax",
    )


def _user_response(user: dict[str, Any], csrf_token: str) -> dict[str, Any]:
    return {"user": user, "csrf_token": csrf_token}


@router.get("/session")
def get_session(request: Request, response: Response) -> dict[str, Any]:
    store = _auth_store(request)
    principal = _principal(request, store, required=False)
    if principal is None:
        _clear_session_cookie(response, request)
        return {"authenticated": False, "user": None, "csrf_token": None, "workspaces": []}
    from manufacturing_rag.auth.workspace_store import WorkspaceStore

    try:
        workspaces = WorkspaceStore(store).list_workspaces(principal.user["id"])
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号或工作区权限服务暂不可用") from exc
    return {
        "authenticated": True,
        "user": principal.user,
        "csrf_token": principal.csrf_token,
        "workspaces": workspaces,
    }


@router.get("/registration")
def registration_status(request: Request) -> dict[str, Any]:
    try:
        return _auth_store(request).registration_status()
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc


@router.post("/register", status_code=201)
def register(payload: RegisterPayload, request: Request, response: Response) -> dict[str, Any]:
    _require_browser_request(request)
    store = _auth_store(request)
    ip_address = _request_ip(request)
    try:
        store.check_registration_rate_limit(ip_address)
        user, token, csrf_token = store.register(
            payload.username,
            payload.display_name,
            payload.password,
            ip_address,
        )
    except LoginTemporarilyLimited as exc:
        raise HTTPException(status_code=429, detail="注册请求过于频繁，请稍后再试") from exc
    except RegistrationClosed as exc:
        try:
            store.record_registration_failure(ip_address)
        except AuthServiceUnavailable as service_exc:
            raise HTTPException(status_code=503, detail="账号服务暂不可用") from service_exc
        raise HTTPException(status_code=403, detail="当前暂未开放注册") from exc
    except AccountAlreadyExists as exc:
        raise HTTPException(status_code=409, detail="该账号已被使用") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc
    _set_session_cookie(response, request, token)
    return _user_response(user, csrf_token)


@router.post("/login")
def login(payload: LoginPayload, request: Request, response: Response) -> dict[str, Any]:
    _require_browser_request(request)
    store = _auth_store(request)
    try:
        user, token, csrf_token = store.login(payload.username, payload.password, _request_ip(request))
    except LoginTemporarilyLimited as exc:
        raise HTTPException(status_code=429, detail="账号或密码错误；尝试过多时请稍后再试") from exc
    except InvalidCredentials as exc:
        raise HTTPException(status_code=401, detail="账号或密码错误") from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc
    _set_session_cookie(response, request, token)
    return _user_response(user, csrf_token)


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    _require_browser_request(request)
    store = _auth_store(request)
    principal = _principal(request, store, required=False)
    if principal:
        _require_csrf(request, principal)
        try:
            store.revoke_session(principal)
        except AuthServiceUnavailable as exc:
            raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc
    _clear_session_cookie(response, request)
    return {"ok": True}


@router.post("/logout-all")
def logout_all(request: Request, response: Response) -> dict[str, bool]:
    _require_browser_request(request)
    store = _auth_store(request)
    principal = _principal(request, store)
    assert principal is not None
    _require_csrf(request, principal)
    try:
        store.revoke_all_sessions(principal.user["id"], principal.user["id"])
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc
    _clear_session_cookie(response, request)
    return {"ok": True}


@router.post("/change-password")
def change_password(payload: ChangePasswordPayload, request: Request, response: Response) -> dict[str, bool]:
    _require_browser_request(request)
    store = _auth_store(request)
    principal = _principal(request, store)
    assert principal is not None
    _require_csrf(request, principal)
    try:
        store.change_password(principal.user["id"], payload.current_password, payload.new_password)
    except CurrentPasswordIncorrect as exc:
        raise HTTPException(status_code=401, detail="当前密码不正确") from exc
    except NewPasswordUnchanged as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AccountUnavailable as exc:
        raise HTTPException(status_code=403, detail="账号当前不可用") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc
    _clear_session_cookie(response, request)
    return {"ok": True}


@router.patch("/admin/registration")
def set_registration_enabled(payload: RegistrationTogglePayload, request: Request) -> dict[str, Any]:
    _require_browser_request(request)
    store = _auth_store(request)
    principal = _principal(request, store)
    assert principal is not None
    _require_csrf(request, principal)
    _require_platform_admin(principal)
    try:
        return store.set_registration_enabled(principal.user["id"], payload.enabled)
    except PlatformAdminRequired as exc:
        raise HTTPException(status_code=403, detail="需要平台管理员权限") from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc


@router.post("/admin/login-limits/clear")
def clear_login_limits(payload: ClearLoginLimitPayload, request: Request) -> dict[str, int]:
    _require_browser_request(request)
    store = _auth_store(request)
    principal = _principal(request, store)
    assert principal is not None
    _require_csrf(request, principal)
    _require_platform_admin(principal)
    if not payload.confirm:
        raise HTTPException(status_code=422, detail="请确认解除该账号的临时登录限制")
    try:
        removed_count = store.clear_login_limits(principal.user["id"], payload.username)
    except PlatformAdminRequired as exc:
        raise HTTPException(status_code=403, detail="需要平台管理员权限") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc
    return {"removed_bucket_count": removed_count}


@router.get("/admin/audit")
def list_platform_audit(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    store = _auth_store(request)
    principal = _principal(request, store)
    assert principal is not None
    _require_platform_admin(principal)
    try:
        return store.list_audit_logs(principal.user["id"], limit, offset)
    except PlatformAdminRequired as exc:
        raise HTTPException(status_code=403, detail="需要平台管理员权限") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号审计数据暂不可用") from exc


@router.get("/admin/users")
def list_platform_users(request: Request, q: str = "", limit: int = 100) -> dict[str, Any]:
    store = _auth_store(request)
    principal = _principal(request, store)
    assert principal is not None
    _require_platform_admin(principal)
    try:
        return {"users": store.list_users(principal.user["id"], q, limit)}
    except PlatformAdminRequired as exc:
        raise HTTPException(status_code=403, detail="需要平台管理员权限") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc


@router.patch("/admin/users/{target_user_id}/platform-role")
def change_platform_role(
    target_user_id: str,
    payload: PlatformRolePayload,
    request: Request,
) -> dict[str, Any]:
    _require_browser_request(request)
    store = _auth_store(request)
    principal = _principal(request, store)
    assert principal is not None
    _require_csrf(request, principal)
    _require_platform_admin(principal)
    try:
        return store.set_platform_role(principal.user["id"], target_user_id, payload.platform_role)
    except PlatformAdminRequired as exc:
        raise HTTPException(status_code=403, detail="需要平台管理员权限") from exc
    except LastPlatformAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc


@router.patch("/admin/users/{target_user_id}/status")
def change_user_status(
    target_user_id: str,
    payload: UserStatusPayload,
    request: Request,
) -> dict[str, Any]:
    _require_browser_request(request)
    store = _auth_store(request)
    principal = _principal(request, store)
    assert principal is not None
    _require_csrf(request, principal)
    _require_platform_admin(principal)
    try:
        return store.set_user_status(principal.user["id"], target_user_id, payload.status)
    except PlatformAdminRequired as exc:
        raise HTTPException(status_code=403, detail="需要平台管理员权限") from exc
    except LastPlatformAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc


@router.post("/admin/users/{target_user_id}/sessions/revoke")
def revoke_user_sessions(target_user_id: str, request: Request) -> dict[str, bool]:
    _require_browser_request(request)
    store = _auth_store(request)
    principal = _principal(request, store)
    assert principal is not None
    _require_csrf(request, principal)
    _require_platform_admin(principal)
    try:
        store.revoke_all_sessions(target_user_id, principal.user["id"], action="admin_revoke_sessions")
        return {"ok": True}
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号服务暂不可用") from exc
