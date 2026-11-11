from __future__ import annotations

from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from manufacturing_rag.auth.routes import (
    _auth_store,
    _principal,
    _require_browser_request,
    _require_csrf,
)
from manufacturing_rag.auth.store import AuthServiceUnavailable, Principal
from manufacturing_rag.auth.workspace_store import (
    WorkspaceConflict,
    WorkspaceForbidden,
    WorkspaceNotFound,
    WorkspaceStore,
)


class StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateWorkspacePayload(StrictPayload):
    name: str = Field(min_length=1, max_length=150)
    description: str = Field(default="", max_length=1500)


class UpdateWorkspacePayload(StrictPayload):
    name: str = Field(min_length=1, max_length=150)
    description: str = Field(default="", max_length=1500)


class MemberRolePayload(StrictPayload):
    role: Literal["admin", "reader", "blocked"] | None


class DepartmentGrantPayload(StrictPayload):
    enabled: bool


class TransferOwnerPayload(StrictPayload):
    new_owner_user_id: str = Field(min_length=1, max_length=36)


class ConfirmPayload(StrictPayload):
    confirm: bool


class CreateDepartmentPayload(StrictPayload):
    name: str = Field(min_length=1, max_length=160)


class UpdateDepartmentPayload(StrictPayload):
    name: str | None = Field(default=None, max_length=160)
    is_active: bool | None = None


class SetUserDepartmentPayload(StrictPayload):
    department_id: str | None


router = APIRouter(prefix="/api/workspaces", tags=["工作区权限"])
admin_router = APIRouter(prefix="/api/admin", tags=["平台管理"])


def _workspace_store(request: Request) -> WorkspaceStore:
    return WorkspaceStore(_auth_store(request))


def _required_principal(request: Request) -> Principal:
    principal = _principal(request, _auth_store(request))
    assert principal is not None
    return principal


def _run(action: Callable[[], Any]) -> Any:
    try:
        return action()
    except WorkspaceNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except WorkspaceForbidden as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except WorkspaceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AuthServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="账号或工作区权限服务暂不可用") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _write_principal(request: Request) -> Principal:
    _require_browser_request(request)
    principal = _required_principal(request)
    _require_csrf(request, principal)
    return principal


@router.get("")
def list_workspaces(request: Request) -> dict[str, Any]:
    principal = _required_principal(request)
    workspaces = _run(lambda: _workspace_store(request).list_workspaces(principal.user["id"]))
    return {"workspaces": workspaces}


@router.post("", status_code=201)
def create_workspace(payload: CreateWorkspacePayload, request: Request) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(
        lambda: _workspace_store(request).create_workspace(
            principal.user["id"], payload.name, payload.description
        )
    )


@router.get("/{workspace_id}")
def get_workspace(workspace_id: str, request: Request) -> dict[str, Any]:
    principal = _required_principal(request)
    return _run(lambda: _workspace_store(request).get_workspace(principal.user["id"], workspace_id))


@router.patch("/{workspace_id}")
def update_workspace(workspace_id: str, payload: UpdateWorkspacePayload, request: Request) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(
        lambda: _workspace_store(request).update_workspace(
            principal.user["id"], workspace_id, payload.name, payload.description
        )
    )


@router.get("/{workspace_id}/members")
def list_members(workspace_id: str, request: Request) -> dict[str, Any]:
    principal = _required_principal(request)
    members = _run(lambda: _workspace_store(request).list_members(principal.user["id"], workspace_id))
    return {"members": members}


@router.put("/{workspace_id}/members/{target_user_id}")
def set_member_role(
    workspace_id: str,
    target_user_id: str,
    payload: MemberRolePayload,
    request: Request,
) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(
        lambda: _workspace_store(request).set_member_role(
            principal.user["id"], workspace_id, target_user_id, payload.role
        )
    )


@router.get("/{workspace_id}/departments")
def list_workspace_departments(workspace_id: str, request: Request) -> dict[str, Any]:
    principal = _required_principal(request)
    departments = _run(
        lambda: _workspace_store(request).list_department_grants(principal.user["id"], workspace_id)
    )
    return {"departments": departments}


@router.put("/{workspace_id}/departments/{department_id}")
def set_workspace_department(
    workspace_id: str,
    department_id: str,
    payload: DepartmentGrantPayload,
    request: Request,
) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(
        lambda: _workspace_store(request).set_department_grant(
            principal.user["id"], workspace_id, department_id, payload.enabled
        )
    )


@router.post("/{workspace_id}/transfer")
def transfer_workspace_owner(workspace_id: str, payload: TransferOwnerPayload, request: Request) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(
        lambda: _workspace_store(request).transfer_owner(
            principal.user["id"], workspace_id, payload.new_owner_user_id
        )
    )


@router.post("/{workspace_id}/archive")
def archive_workspace(workspace_id: str, payload: ConfirmPayload, request: Request) -> dict[str, Any]:
    principal = _write_principal(request)
    if not payload.confirm:
        raise HTTPException(status_code=422, detail="请确认归档该工作区")
    return _run(lambda: _workspace_store(request).set_archived(principal.user["id"], workspace_id, True))


@router.post("/{workspace_id}/restore")
def restore_workspace(workspace_id: str, payload: ConfirmPayload, request: Request) -> dict[str, Any]:
    principal = _write_principal(request)
    if not payload.confirm:
        raise HTTPException(status_code=422, detail="请确认恢复该工作区")
    return _run(lambda: _workspace_store(request).set_archived(principal.user["id"], workspace_id, False))


@admin_router.get("/departments")
def list_departments(request: Request) -> dict[str, Any]:
    principal = _required_principal(request)
    departments = _run(lambda: _workspace_store(request).list_departments(principal.user["id"]))
    return {"departments": departments}


@admin_router.post("/departments", status_code=201)
def create_department(payload: CreateDepartmentPayload, request: Request) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(lambda: _workspace_store(request).create_department(principal.user["id"], payload.name))


@admin_router.patch("/departments/{department_id}")
def update_department(
    department_id: str,
    payload: UpdateDepartmentPayload,
    request: Request,
) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(
        lambda: _workspace_store(request).update_department(
            principal.user["id"], department_id, payload.name, payload.is_active
        )
    )


@admin_router.patch("/users/{user_id}/department")
def set_user_department(user_id: str, payload: SetUserDepartmentPayload, request: Request) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(
        lambda: _workspace_store(request).set_user_department(
            principal.user["id"], user_id, payload.department_id
        )
    )
