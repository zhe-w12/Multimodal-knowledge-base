from __future__ import annotations

from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Query, Request
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
    expected_workspace_version: int | None = Field(default=None, ge=1)


class DepartmentGrantPayload(StrictPayload):
    enabled: bool
    expected_workspace_version: int | None = Field(default=None, ge=1)


class TransferOwnerPayload(StrictPayload):
    new_owner_user_id: str = Field(min_length=1, max_length=36)


class ConfirmPayload(StrictPayload):
    confirm: bool


class DeleteWorkspacePayload(StrictPayload):
    confirm: bool
    workspace_name: str = Field(min_length=1, max_length=150)


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


@router.get("/{workspace_id}/audit")
def list_workspace_audit(
    workspace_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    principal = _required_principal(request)
    return _run(
        lambda: _workspace_store(request).list_workspace_audit(
            principal.user["id"], workspace_id, limit, offset
        )
    )


@router.get("/{workspace_id}/members")
def list_members(workspace_id: str, request: Request) -> dict[str, Any]:
    principal = _required_principal(request)
    members = _run(lambda: _workspace_store(request).list_members(principal.user["id"], workspace_id))
    return {"members": members}


@router.get("/{workspace_id}/users/search")
def find_member_candidate(workspace_id: str, username: str, request: Request) -> dict[str, Any]:
    principal = _required_principal(request)
    user = _run(
        lambda: _workspace_store(request).find_workspace_member_candidate(
            principal.user["id"], workspace_id, username
        )
    )
    return {"user": user}


@router.post("/{workspace_id}/members/{target_user_id}/preview")
def preview_member_role(
    workspace_id: str,
    target_user_id: str,
    payload: MemberRolePayload,
    request: Request,
) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(
        lambda: _workspace_store(request).preview_member_role(
            principal.user["id"],
            workspace_id,
            target_user_id,
            payload.role,
            payload.expected_workspace_version,
        )
    )


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
            principal.user["id"], workspace_id, target_user_id, payload.role, payload.expected_workspace_version
        )
    )


@router.get("/{workspace_id}/departments")
def list_workspace_departments(workspace_id: str, request: Request) -> dict[str, Any]:
    principal = _required_principal(request)
    departments = _run(
        lambda: _workspace_store(request).list_department_grants(principal.user["id"], workspace_id)
    )
    return {"departments": departments}


@router.get("/{workspace_id}/department-options")
def list_workspace_department_options(workspace_id: str, request: Request) -> dict[str, Any]:
    principal = _required_principal(request)
    departments = _run(
        lambda: _workspace_store(request).list_workspace_department_options(
            principal.user["id"], workspace_id
        )
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
            principal.user["id"], workspace_id, department_id, payload.enabled, payload.expected_workspace_version
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


@router.delete("/{workspace_id}")
def delete_workspace(
    workspace_id: str,
    payload: DeleteWorkspacePayload,
    request: Request,
) -> dict[str, Any]:
    principal = _write_principal(request)
    if not payload.confirm:
        raise HTTPException(status_code=422, detail="请确认永久删除该工作区")
    lock_factory = getattr(request.app.state, "workspace_operation_lock", None)
    if lock_factory is None:
        raise HTTPException(status_code=503, detail="工作区维护服务暂不可用")
    operation_lock = lock_factory(workspace_id)
    operation_lock.acquire()
    try:
        store = _workspace_store(request)
        _run(
            lambda: store.begin_workspace_deletion(
                principal.user["id"], workspace_id, payload.workspace_name
            )
        )

        jobs = getattr(request.app.state, "document_delete_jobs", {})
        jobs_lock = getattr(request.app.state, "document_delete_jobs_lock", None)
        if jobs_lock is not None:
            with jobs_lock:
                pending_job = any(
                    job.get("workspace_id") == workspace_id
                    and job.get("status") in {"queued", "running"}
                    for job in jobs.values()
                )
        else:
            pending_job = any(
                job.get("workspace_id") == workspace_id
                and job.get("status") in {"queued", "running"}
                for job in jobs.values()
            )
        if pending_job:
            raise HTTPException(status_code=409, detail="该工作区仍有资料删除任务运行，任务完成后请重试")

        delete_data = getattr(request.app.state, "delete_workspace_data", None)
        if delete_data is None:
            raise HTTPException(status_code=503, detail="工作区资料清理服务暂不可用")
        cleanup_result = delete_data(workspace_id)
        deletion_result = _run(lambda: store.complete_workspace_deletion(principal.user["id"], workspace_id))
        return {"ok": True, **cleanup_result, **deletion_result}
    finally:
        operation_lock.release()


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


@admin_router.post("/users/{user_id}/department/preview")
def preview_user_department(user_id: str, payload: SetUserDepartmentPayload, request: Request) -> dict[str, Any]:
    principal = _write_principal(request)
    return _run(
        lambda: _workspace_store(request).preview_user_department(
            principal.user["id"], user_id, payload.department_id
        )
    )
