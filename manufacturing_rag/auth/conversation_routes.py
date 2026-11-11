from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from manufacturing_rag.auth.conversation_store import ConversationLocked, ConversationNotFound, ConversationStore
from manufacturing_rag.auth.routes import _auth_store, _principal, _require_browser_request, _require_csrf
from manufacturing_rag.auth.store import AuthServiceUnavailable
from manufacturing_rag.auth.workspace_store import WorkspaceForbidden, WorkspaceNotFound, WorkspaceStore


router = APIRouter(prefix="/api/conversations", tags=["个人对话"])


class StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateConversationPayload(StrictPayload):
    scope_type: str = Field(pattern="^(all|common|workspace)$")
    workspace_id: str | None = Field(default=None, max_length=128)


class RenameConversationPayload(StrictPayload):
    title: str = Field(min_length=1, max_length=120)


def _readable_workspace_ids(store, user_id: str) -> list[str]:
    try:
        return WorkspaceStore(store).resolve_readable_scope(user_id, "all")["workspace_ids"]
    except (WorkspaceForbidden, WorkspaceNotFound, AuthServiceUnavailable) as exc:
        raise HTTPException(status_code=503, detail="账号或工作区权限服务暂不可用") from exc


def _map_store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ConversationNotFound):
        return HTTPException(status_code=404, detail="对话不存在")
    if isinstance(exc, ConversationLocked):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, AuthServiceUnavailable):
        return HTTPException(status_code=503, detail="对话数据服务暂不可用")
    return HTTPException(status_code=500, detail="对话服务暂不可用")


@router.get("")
def list_conversations(request: Request) -> dict[str, Any]:
    auth = _auth_store(request)
    principal = _principal(request, auth)
    try:
        return {"conversations": ConversationStore(auth).list(principal.user["id"])}
    except Exception as exc:  # noqa: BLE001 - keep storage details private.
        raise _map_store_error(exc) from exc


@router.post("", status_code=201)
def create_conversation(payload: CreateConversationPayload, request: Request) -> dict[str, Any]:
    _require_browser_request(request)
    auth = _auth_store(request)
    principal = _principal(request, auth)
    _require_csrf(request, principal)
    user_id = principal.user["id"]
    workspace_id = payload.workspace_id if payload.scope_type == "workspace" else None
    scope_type = payload.scope_type
    try:
        target_scope = "workspace" if scope_type == "workspace" else scope_type
        WorkspaceStore(auth).resolve_readable_scope(user_id, target_scope, workspace_id)
        conversation = ConversationStore(auth).create(user_id, scope_type, workspace_id)
        return {"conversation": conversation}
    except (WorkspaceNotFound, WorkspaceForbidden) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, WorkspaceNotFound) else 403, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - keep storage details private.
        raise _map_store_error(exc) from exc


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
    auth = _auth_store(request)
    principal = _principal(request, auth)
    try:
        readable = _readable_workspace_ids(auth, principal.user["id"])
        conversation = ConversationStore(auth).detail(principal.user["id"], conversation_id, readable)
        return {"conversation": conversation}
    except Exception as exc:  # noqa: BLE001 - keep storage details private.
        raise _map_store_error(exc) from exc


@router.patch("/{conversation_id}")
def rename_conversation(conversation_id: str, payload: RenameConversationPayload, request: Request) -> dict[str, Any]:
    _require_browser_request(request)
    auth = _auth_store(request)
    principal = _principal(request, auth)
    _require_csrf(request, principal)
    try:
        conversation = ConversationStore(auth).rename(principal.user["id"], conversation_id, payload.title)
        return {"conversation": conversation}
    except Exception as exc:  # noqa: BLE001 - keep storage details private.
        raise _map_store_error(exc) from exc


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request) -> dict[str, bool]:
    _require_browser_request(request, require_json=False)
    auth = _auth_store(request)
    principal = _principal(request, auth)
    _require_csrf(request, principal)
    try:
        ConversationStore(auth).delete(principal.user["id"], conversation_id)
        return {"deleted": True}
    except Exception as exc:  # noqa: BLE001 - keep storage details private.
        raise _map_store_error(exc) from exc
