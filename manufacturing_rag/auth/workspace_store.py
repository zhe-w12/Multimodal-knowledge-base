from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from manufacturing_rag.auth.models import (
    Department,
    DepartmentWorkspaceGrant,
    User,
    Workspace,
    WorkspaceMembership,
    utcnow,
)
from manufacturing_rag.auth.store import AuthServiceUnavailable, AuthStore


class WorkspaceNotFound(RuntimeError):
    pass


class WorkspaceForbidden(RuntimeError):
    pass


class WorkspaceConflict(RuntimeError):
    pass


F = TypeVar("F", bound=Callable[..., Any])


def _database_operation(function: F) -> F:
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("工作区权限数据暂不可用") from exc

    return wrapped  # type: ignore[return-value]


class WorkspaceStore:
    """Persistent workspace permissions backed by the account database."""

    def __init__(self, auth_store: AuthStore):
        self.auth = auth_store

    @staticmethod
    def _workspace_dict(workspace: Workspace) -> dict[str, Any]:
        return {
            "id": workspace.id,
            "name": workspace.name,
            "description": workspace.description,
            "workspace_type": workspace.workspace_type,
            "status": workspace.status,
            "created_by": workspace.created_by,
            "owner_user_id": workspace.owner_user_id,
            "version": workspace.version,
            "created_at": workspace.created_at.isoformat(),
            "updated_at": workspace.updated_at.isoformat(),
        }

    @staticmethod
    def _load_user(db, user_id: str) -> User:
        user = db.get(User, user_id)
        if user is None:
            raise WorkspaceForbidden("账号不可用")
        if user.status != "active":
            raise WorkspaceForbidden("账号不可用")
        return user

    @staticmethod
    def _access_for(
        user: User,
        workspace: Workspace,
        direct_role: str | None,
        department_granted: bool,
    ) -> dict[str, Any]:
        is_platform_admin = user.platform_role == "platform_admin"
        is_owner = workspace.workspace_type == "authorized" and workspace.owner_user_id == user.id
        is_active = workspace.status == "active"

        if workspace.workspace_type == "general":
            can_read = is_active
            can_manage = is_platform_admin
            role = "platform_admin" if is_platform_admin else ("reader" if can_read else None)
        elif is_platform_admin:
            can_read = is_active
            can_manage = True
            role = "platform_admin"
        elif is_owner:
            can_read = is_active
            can_manage = True
            role = "owner"
        elif direct_role == "admin":
            can_read = is_active
            can_manage = True
            role = "admin"
        elif direct_role == "blocked":
            can_read = False
            can_manage = False
            role = "blocked"
        elif direct_role == "reader":
            can_read = is_active
            can_manage = False
            role = "reader"
        elif department_granted:
            can_read = is_active
            can_manage = False
            role = "department_reader"
        else:
            can_read = False
            can_manage = False
            role = None

        return {
            "role": role,
            "can_read": can_read,
            "can_manage": can_manage,
            "can_manage_members": can_manage and workspace.workspace_type == "authorized",
            "can_manage_content": can_manage and is_active,
            "is_owner": is_owner,
            "is_platform_admin": is_platform_admin,
        }

    @classmethod
    def _access_in_session(cls, db, user: User, workspace: Workspace) -> dict[str, Any]:
        direct_role = db.scalar(
            select(WorkspaceMembership.role).where(
                WorkspaceMembership.workspace_id == workspace.id,
                WorkspaceMembership.user_id == user.id,
            )
        )
        department_granted = False
        if user.department_id:
            department = db.get(Department, user.department_id)
            if department is not None and department.is_active:
                department_granted = bool(
                    db.scalar(
                        select(DepartmentWorkspaceGrant.id).where(
                            DepartmentWorkspaceGrant.department_id == department.id,
                            DepartmentWorkspaceGrant.workspace_id == workspace.id,
                        )
                    )
                )
        return cls._access_for(user, workspace, direct_role, department_granted)

    @staticmethod
    def _workspace_or_404(db, workspace_id: str, *, lock: bool = False) -> Workspace:
        statement = select(Workspace).where(Workspace.id == workspace_id)
        if lock:
            statement = statement.with_for_update()
        workspace = db.scalar(statement)
        if workspace is None:
            raise WorkspaceNotFound("工作区不存在")
        return workspace

    @classmethod
    def _require_access(cls, db, user: User, workspace: Workspace, *, manage: bool = False) -> dict[str, Any]:
        access = cls._access_in_session(db, user, workspace)
        allowed = access["can_manage"] if manage else access["can_read"]
        if not allowed:
            raise WorkspaceForbidden("无权访问该工作区")
        return access

    @staticmethod
    def _require_owner_or_platform(user: User, workspace: Workspace) -> None:
        if user.platform_role == "platform_admin" or workspace.owner_user_id == user.id:
            return
        raise WorkspaceForbidden("仅本区负责人或平台管理员可以执行此操作")

    @staticmethod
    def _require_active_authorized(workspace: Workspace) -> None:
        if workspace.workspace_type != "authorized":
            raise WorkspaceForbidden("通用工作区不能执行该操作")
        if workspace.status != "active":
            raise WorkspaceConflict("工作区已归档，请先恢复")

    def _audit(self, db, actor_id: str, action: str, workspace: Workspace, details: dict[str, Any]) -> None:
        self.auth._audit(db, actor_id, action, "workspace", workspace.id, details)

    @staticmethod
    def _bump_user(db, user_id: str) -> None:
        user = db.get(User, user_id)
        if user is not None:
            user.permissions_version += 1

    @staticmethod
    def _validate_name_and_description(name: str, description: str) -> tuple[str, str]:
        normalized_name = name.strip()
        normalized_description = description.strip()
        if not 1 <= len(normalized_name) <= 50:
            raise ValueError("工作区名称须为 1～50 个字符")
        if len(normalized_description) > 500:
            raise ValueError("工作区说明不能超过 500 个字符")
        return normalized_name, normalized_description

    @_database_operation
    def list_workspaces(self, user_id: str) -> list[dict[str, Any]]:
        self.auth.ensure_schema()
        with self.auth.Session() as db:
            user = self._load_user(db, user_id)
            workspaces = db.scalars(
                select(Workspace).order_by(Workspace.workspace_type, Workspace.name, Workspace.id)
            ).all()
            visible = []
            for workspace in workspaces:
                access = self._access_in_session(db, user, workspace)
                if access["can_read"] or access["can_manage"]:
                    visible.append({**self._workspace_dict(workspace), **access})
            return visible

    @_database_operation
    def resolve_readable_scope(
        self,
        user_id: str,
        scope: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a browser-provided scope to server-authorized workspace IDs."""

        self.auth.ensure_schema()
        if scope not in {"all", "common", "workspace"}:
            raise ValueError("检索范围无效")
        with self.auth.Session() as db:
            user = self._load_user(db, user_id)
            workspaces = db.scalars(select(Workspace).order_by(Workspace.id)).all()
            selected: list[Workspace] = []
            if scope == "all":
                selected = workspaces
            elif scope == "common":
                selected = [workspace for workspace in workspaces if workspace.general_marker == "general"]
            else:
                resolved_id = workspace_id or "campus-default"
                selected = [workspace for workspace in workspaces if workspace.id == resolved_id]
                if not selected:
                    raise WorkspaceNotFound("工作区不存在")

            resolved = []
            for workspace in selected:
                access = self._access_in_session(db, user, workspace)
                if access["can_read"]:
                    resolved.append({**self._workspace_dict(workspace), **access})
                elif scope != "all":
                    raise WorkspaceForbidden("无权检索该工作区")
            return {
                "user_id": user.id,
                "user_permissions_version": user.permissions_version,
                "workspaces": resolved,
                "workspace_ids": [workspace["id"] for workspace in resolved],
                "cache_scope": ":".join(
                    [f"u{user.permissions_version}"]
                    + [f"{workspace['id']}={workspace['version']}:{workspace['role']}" for workspace in resolved]
                ),
            }

    @_database_operation
    def create_workspace(self, actor_user_id: str, name: str, description: str = "") -> dict[str, Any]:
        self.auth.ensure_schema()
        normalized_name, normalized_description = self._validate_name_and_description(name, description)
        workspace_id = str(uuid4())
        try:
            with self.auth.Session.begin() as db:
                actor = self._load_user(db, actor_user_id)
                workspace = Workspace(
                    id=workspace_id,
                    name=normalized_name,
                    description=normalized_description,
                    workspace_type="authorized",
                    general_marker=None,
                    status="active",
                    created_by=actor.id,
                    owner_user_id=actor.id,
                    version=1,
                )
                db.add(workspace)
                db.flush()
                db.add(
                    WorkspaceMembership(
                        id=str(uuid4()),
                        workspace_id=workspace.id,
                        user_id=actor.id,
                        role="admin",
                        granted_by=actor.id,
                    )
                )
                self._audit(db, actor.id, "workspace_created", workspace, {"name": normalized_name})
            return {**self._workspace_dict(workspace), **self._access_for(actor, workspace, "admin", False)}
        except IntegrityError as exc:
            raise WorkspaceConflict("工作区创建冲突，请重试") from exc

    @_database_operation
    def get_workspace(self, user_id: str, workspace_id: str) -> dict[str, Any]:
        self.auth.ensure_schema()
        with self.auth.Session() as db:
            user = self._load_user(db, user_id)
            workspace = self._workspace_or_404(db, workspace_id)
            access = self._access_in_session(db, user, workspace)
            if not (access["can_read"] or access["can_manage"]):
                raise WorkspaceForbidden("无权访问该工作区")
            return {**self._workspace_dict(workspace), **access}

    @_database_operation
    def update_workspace(self, actor_user_id: str, workspace_id: str, name: str, description: str) -> dict[str, Any]:
        self.auth.ensure_schema()
        normalized_name, normalized_description = self._validate_name_and_description(name, description)
        with self.auth.Session.begin() as db:
            actor = self._load_user(db, actor_user_id)
            workspace = self._workspace_or_404(db, workspace_id, lock=True)
            if workspace.workspace_type != "authorized":
                raise WorkspaceForbidden("通用工作区设置仅平台管理员可维护")
            self._require_owner_or_platform(actor, workspace)
            workspace.name = normalized_name
            workspace.description = normalized_description
            workspace.version += 1
            workspace.updated_at = utcnow()
            self._audit(db, actor.id, "workspace_updated", workspace, {"name": normalized_name})
            return {**self._workspace_dict(workspace), **self._access_in_session(db, actor, workspace)}

    @_database_operation
    def list_members(self, actor_user_id: str, workspace_id: str) -> list[dict[str, Any]]:
        self.auth.ensure_schema()
        with self.auth.Session() as db:
            actor = self._load_user(db, actor_user_id)
            workspace = self._workspace_or_404(db, workspace_id)
            self._require_access(db, actor, workspace, manage=True)
            rows = db.execute(
                select(WorkspaceMembership, User)
                .join(User, User.id == WorkspaceMembership.user_id)
                .where(WorkspaceMembership.workspace_id == workspace.id)
                .order_by(User.username)
            ).all()
            result = []
            for membership, user in rows:
                result.append(
                    {
                        "user_id": user.id,
                        "username": user.username,
                        "display_name": user.display_name,
                        "department_id": user.department_id,
                        "status": user.status,
                        "role": membership.role,
                        "is_owner": user.id == workspace.owner_user_id,
                    }
                )
            return result

    @_database_operation
    def find_workspace_member_candidate(
        self,
        actor_user_id: str,
        workspace_id: str,
        username: str,
    ) -> dict[str, Any] | None:
        normalized_username = username.strip().casefold()
        if not normalized_username:
            raise ValueError("请输入准确账号")
        self.auth.ensure_schema()
        with self.auth.Session() as db:
            actor = self._load_user(db, actor_user_id)
            workspace = self._workspace_or_404(db, workspace_id)
            self._require_access(db, actor, workspace, manage=True)
            self._require_active_authorized(workspace)
            user = db.scalar(
                select(User).where(
                    User.username == normalized_username,
                    User.status == "active",
                )
            )
            if user is None:
                return None
            return {
                "user_id": user.id,
                "username": user.username,
                "display_name": user.display_name,
                "department_id": user.department_id,
            }

    @_database_operation
    def _active_admin_ids(self, db, workspace: Workspace) -> set[str]:
        admin_ids = set(
            db.scalars(
                select(WorkspaceMembership.user_id)
                .join(User, User.id == WorkspaceMembership.user_id)
                .where(
                    WorkspaceMembership.workspace_id == workspace.id,
                    WorkspaceMembership.role == "admin",
                    User.status == "active",
                )
            ).all()
        )
        if workspace.owner_user_id:
            owner = db.get(User, workspace.owner_user_id)
            if owner is not None and owner.status == "active":
                admin_ids.add(owner.id)
        return admin_ids

    @_database_operation
    def set_member_role(
        self,
        actor_user_id: str,
        workspace_id: str,
        target_user_id: str,
        role: str | None,
    ) -> dict[str, Any]:
        if role not in {None, "reader", "admin", "blocked"}:
            raise ValueError("成员角色无效")
        self.auth.ensure_schema()
        try:
            with self.auth.Session.begin() as db:
                actor = self._load_user(db, actor_user_id)
                workspace = self._workspace_or_404(db, workspace_id, lock=True)
                self._require_active_authorized(workspace)
                self._require_access(db, actor, workspace, manage=True)
                target = self._load_user(db, target_user_id)
                is_platform_admin = actor.platform_role == "platform_admin"
                is_owner = actor.id == workspace.owner_user_id
                membership = db.scalar(
                    select(WorkspaceMembership)
                    .where(
                        WorkspaceMembership.workspace_id == workspace.id,
                        WorkspaceMembership.user_id == target.id,
                    )
                    .with_for_update()
                )
                old_role = membership.role if membership else None
                admin_assignment = role == "admin" or old_role == "admin"
                if admin_assignment and not (is_platform_admin or is_owner):
                    raise WorkspaceForbidden("只有本区负责人或平台管理员可以增加、撤销本区管理员")
                if target.id == workspace.owner_user_id and role != "admin":
                    raise WorkspaceConflict("请先转交工作区负责人，不能移除或禁止负责人")

                if old_role == "admin" and role != "admin":
                    active_admins = self._active_admin_ids(db, workspace)
                    if target.id in active_admins and len(active_admins - {target.id}) < 1:
                        raise WorkspaceConflict("工作区至少保留一名启用管理员")

                if role is None:
                    if membership is not None:
                        db.delete(membership)
                elif membership is None:
                    db.add(
                        WorkspaceMembership(
                            id=str(uuid4()),
                            workspace_id=workspace.id,
                            user_id=target.id,
                            role=role,
                            granted_by=actor.id,
                        )
                    )
                else:
                    membership.role = role
                    membership.granted_by = actor.id
                    membership.updated_at = utcnow()

                workspace.version += 1
                workspace.updated_at = utcnow()
                self._bump_user(db, target.id)
                self._audit(
                    db,
                    actor.id,
                    "workspace_member_changed",
                    workspace,
                    {"target_user_id": target.id, "old_role": old_role, "new_role": role},
                )
                return {
                    "workspace_id": workspace.id,
                    "user_id": target.id,
                    "role": role,
                    "workspace_version": workspace.version,
                    "is_owner": target.id == workspace.owner_user_id,
                }
        except IntegrityError as exc:
            raise WorkspaceConflict("成员授权冲突，请重试") from exc

    @_database_operation
    def list_department_grants(self, actor_user_id: str, workspace_id: str) -> list[dict[str, Any]]:
        self.auth.ensure_schema()
        with self.auth.Session() as db:
            actor = self._load_user(db, actor_user_id)
            workspace = self._workspace_or_404(db, workspace_id)
            self._require_access(db, actor, workspace, manage=True)
            rows = db.execute(
                select(DepartmentWorkspaceGrant, Department)
                .join(Department, Department.id == DepartmentWorkspaceGrant.department_id)
                .where(DepartmentWorkspaceGrant.workspace_id == workspace.id)
                .order_by(Department.name)
            ).all()
            return [
                {
                    "department_id": department.id,
                    "name": department.name,
                    "is_active": department.is_active,
                }
                for _grant, department in rows
            ]

    @_database_operation
    def list_workspace_department_options(self, actor_user_id: str, workspace_id: str) -> list[dict[str, Any]]:
        self.auth.ensure_schema()
        with self.auth.Session() as db:
            actor = self._load_user(db, actor_user_id)
            workspace = self._workspace_or_404(db, workspace_id)
            self._require_access(db, actor, workspace, manage=True)
            grants = set(
                db.scalars(
                    select(DepartmentWorkspaceGrant.department_id).where(
                        DepartmentWorkspaceGrant.workspace_id == workspace.id
                    )
                ).all()
            )
            departments = db.scalars(select(Department).order_by(Department.name)).all()
            return [
                {
                    "department_id": department.id,
                    "name": department.name,
                    "is_active": department.is_active,
                    "enabled": department.id in grants,
                }
                for department in departments
            ]

    @_database_operation
    def set_department_grant(
        self,
        actor_user_id: str,
        workspace_id: str,
        department_id: str,
        enabled: bool,
    ) -> dict[str, Any]:
        self.auth.ensure_schema()
        try:
            with self.auth.Session.begin() as db:
                actor = self._load_user(db, actor_user_id)
                workspace = self._workspace_or_404(db, workspace_id, lock=True)
                self._require_active_authorized(workspace)
                self._require_access(db, actor, workspace, manage=True)
                department = db.get(Department, department_id)
                if department is None:
                    raise WorkspaceNotFound("部门不存在")
                grant = db.scalar(
                    select(DepartmentWorkspaceGrant)
                    .where(
                        DepartmentWorkspaceGrant.department_id == department.id,
                        DepartmentWorkspaceGrant.workspace_id == workspace.id,
                    )
                    .with_for_update()
                )
                if enabled:
                    if not department.is_active:
                        raise WorkspaceConflict("部门已停用，不能授予工作区访问权限")
                    if grant is None:
                        db.add(
                            DepartmentWorkspaceGrant(
                                id=str(uuid4()),
                                department_id=department.id,
                                workspace_id=workspace.id,
                                granted_by=actor.id,
                            )
                        )
                elif grant is not None:
                    db.delete(grant)
                workspace.version += 1
                workspace.updated_at = utcnow()
                for user in db.scalars(select(User).where(User.department_id == department.id)).all():
                    user.permissions_version += 1
                self._audit(
                    db,
                    actor.id,
                    "workspace_department_grant_changed",
                    workspace,
                    {"department_id": department.id, "enabled": enabled},
                )
                return {
                    "workspace_id": workspace.id,
                    "department_id": department.id,
                    "enabled": enabled,
                    "workspace_version": workspace.version,
                }
        except IntegrityError as exc:
            raise WorkspaceConflict("部门授权冲突，请重试") from exc

    @_database_operation
    def transfer_owner(self, actor_user_id: str, workspace_id: str, new_owner_user_id: str) -> dict[str, Any]:
        self.auth.ensure_schema()
        try:
            with self.auth.Session.begin() as db:
                actor = self._load_user(db, actor_user_id)
                workspace = self._workspace_or_404(db, workspace_id, lock=True)
                self._require_active_authorized(workspace)
                self._require_owner_or_platform(actor, workspace)
                new_owner = self._load_user(db, new_owner_user_id)
                if new_owner.id == workspace.owner_user_id:
                    raise WorkspaceConflict("该用户已经是本区负责人")
                membership = db.scalar(
                    select(WorkspaceMembership)
                    .where(
                        WorkspaceMembership.workspace_id == workspace.id,
                        WorkspaceMembership.user_id == new_owner.id,
                    )
                    .with_for_update()
                )
                if membership is None:
                    db.add(
                        WorkspaceMembership(
                            id=str(uuid4()),
                            workspace_id=workspace.id,
                            user_id=new_owner.id,
                            role="admin",
                            granted_by=actor.id,
                        )
                    )
                else:
                    membership.role = "admin"
                    membership.granted_by = actor.id
                    membership.updated_at = utcnow()
                old_owner_id = workspace.owner_user_id
                workspace.owner_user_id = new_owner.id
                workspace.version += 1
                workspace.updated_at = utcnow()
                if old_owner_id:
                    self._bump_user(db, old_owner_id)
                self._bump_user(db, new_owner.id)
                self._audit(
                    db,
                    actor.id,
                    "workspace_owner_transferred",
                    workspace,
                    {"old_owner_user_id": old_owner_id, "new_owner_user_id": new_owner.id},
                )
                return {**self._workspace_dict(workspace), "owner_user_id": new_owner.id}
        except IntegrityError as exc:
            raise WorkspaceConflict("负责人转交冲突，请重试") from exc

    @_database_operation
    def set_archived(self, actor_user_id: str, workspace_id: str, archived: bool) -> dict[str, Any]:
        self.auth.ensure_schema()
        with self.auth.Session.begin() as db:
            actor = self._load_user(db, actor_user_id)
            workspace = self._workspace_or_404(db, workspace_id, lock=True)
            if workspace.workspace_type != "authorized":
                raise WorkspaceForbidden("通用工作区不能归档")
            self._require_owner_or_platform(actor, workspace)
            target_status = "archived" if archived else "active"
            if workspace.status == target_status:
                raise WorkspaceConflict("工作区状态未发生变化")
            workspace.status = target_status
            workspace.version += 1
            workspace.updated_at = utcnow()
            self._bump_user(db, workspace.owner_user_id or "")
            self._audit(db, actor.id, "workspace_archived" if archived else "workspace_restored", workspace, {})
            return {**self._workspace_dict(workspace), **self._access_in_session(db, actor, workspace)}

    @_database_operation
    def list_departments(self, actor_user_id: str) -> list[dict[str, Any]]:
        self.auth.ensure_schema()
        with self.auth.Session() as db:
            actor = self._load_user(db, actor_user_id)
            if actor.platform_role != "platform_admin":
                raise WorkspaceForbidden("需要平台管理员权限")
            departments = db.scalars(select(Department).order_by(Department.name)).all()
            return [
                {
                    "id": department.id,
                    "name": department.name,
                    "is_active": department.is_active,
                    "version": department.version,
                    "active_user_count": int(
                        db.scalar(
                            select(func.count(User.id)).where(
                                User.department_id == department.id,
                                User.status == "active",
                            )
                        )
                        or 0
                    ),
                }
                for department in departments
            ]

    @_database_operation
    def create_department(self, actor_user_id: str, name: str) -> dict[str, Any]:
        normalized_name = name.strip()
        if not 1 <= len(normalized_name) <= 80:
            raise ValueError("部门名称须为 1～80 个字符")
        self.auth.ensure_schema()
        try:
            with self.auth.Session.begin() as db:
                actor = self._load_user(db, actor_user_id)
                if actor.platform_role != "platform_admin":
                    raise WorkspaceForbidden("需要平台管理员权限")
                department = Department(id=str(uuid4()), name=normalized_name, is_active=True, version=1)
                db.add(department)
                db.flush()
                self.auth._audit(db, actor.id, "department_created", "department", department.id, {"name": department.name})
                return {"id": department.id, "name": department.name, "is_active": True, "version": department.version}
        except IntegrityError as exc:
            raise WorkspaceConflict("部门名称已存在") from exc

    @_database_operation
    def update_department(
        self,
        actor_user_id: str,
        department_id: str,
        name: str | None = None,
        is_active: bool | None = None,
    ) -> dict[str, Any]:
        normalized_name = name.strip() if name is not None else None
        if normalized_name is not None and not 1 <= len(normalized_name) <= 80:
            raise ValueError("部门名称须为 1～80 个字符")
        if name is None and is_active is None:
            raise ValueError("请提供需要更新的部门字段")
        self.auth.ensure_schema()
        try:
            with self.auth.Session.begin() as db:
                actor = self._load_user(db, actor_user_id)
                if actor.platform_role != "platform_admin":
                    raise WorkspaceForbidden("需要平台管理员权限")
                department = db.get(Department, department_id)
                if department is None:
                    raise WorkspaceNotFound("部门不存在")
                if is_active is False:
                    member_count = db.scalar(
                        select(User.id).where(User.department_id == department.id, User.status == "active").limit(1)
                    )
                    if member_count:
                        raise WorkspaceConflict("请先调整该部门的启用成员，再停用部门")
                if normalized_name is not None:
                    department.name = normalized_name
                if is_active is not None:
                    department.is_active = is_active
                department.version += 1
                department.updated_at = utcnow()
                for user in db.scalars(select(User).where(User.department_id == department.id)).all():
                    user.permissions_version += 1
                self.auth._audit(
                    db,
                    actor.id,
                    "department_updated",
                    "department",
                    department.id,
                    {"name": department.name, "is_active": department.is_active},
                )
                return {
                    "id": department.id,
                    "name": department.name,
                    "is_active": department.is_active,
                    "version": department.version,
                }
        except IntegrityError as exc:
            raise WorkspaceConflict("部门名称已存在") from exc

    @_database_operation
    def set_user_department(self, actor_user_id: str, user_id: str, department_id: str | None) -> dict[str, Any]:
        self.auth.ensure_schema()
        with self.auth.Session.begin() as db:
            actor = self._load_user(db, actor_user_id)
            if actor.platform_role != "platform_admin":
                raise WorkspaceForbidden("需要平台管理员权限")
            user = db.get(User, user_id)
            if user is None:
                raise WorkspaceNotFound("账号不存在")
            if department_id is not None:
                department = db.get(Department, department_id)
                if department is None:
                    raise WorkspaceNotFound("部门不存在")
                if not department.is_active:
                    raise WorkspaceConflict("不能将账号分配到已停用部门")
            old_department_id = user.department_id
            user.department_id = department_id
            user.permissions_version += 1
            self.auth._audit(
                db,
                actor.id,
                "user_department_changed",
                "user",
                user.id,
                {"old_department_id": old_department_id, "new_department_id": department_id},
            )
            return {
                "user_id": user.id,
                "department_id": user.department_id,
                "permissions_version": user.permissions_version,
            }

    @_database_operation
    def preview_user_department(
        self,
        actor_user_id: str,
        user_id: str,
        department_id: str | None,
    ) -> dict[str, Any]:
        self.auth.ensure_schema()
        with self.auth.Session() as db:
            actor = self._load_user(db, actor_user_id)
            if actor.platform_role != "platform_admin":
                raise WorkspaceForbidden("需要平台管理员权限")
            user = self._load_user(db, user_id)
            next_department = None
            if department_id is not None:
                next_department = db.get(Department, department_id)
                if next_department is None:
                    raise WorkspaceNotFound("部门不存在")
                if not next_department.is_active:
                    raise WorkspaceConflict("不能将账号分配到已停用部门")
            current_department = db.get(Department, user.department_id) if user.department_id else None

            workspaces = db.scalars(select(Workspace).order_by(Workspace.name, Workspace.id)).all()
            changes = []
            for workspace in workspaces:
                before = self._access_in_session(db, user, workspace)
                direct_role = db.scalar(
                    select(WorkspaceMembership.role).where(
                        WorkspaceMembership.workspace_id == workspace.id,
                        WorkspaceMembership.user_id == user.id,
                    )
                )
                department_granted = False
                if next_department is not None:
                    department_granted = bool(
                        db.scalar(
                            select(DepartmentWorkspaceGrant.id).where(
                                DepartmentWorkspaceGrant.department_id == next_department.id,
                                DepartmentWorkspaceGrant.workspace_id == workspace.id,
                            )
                        )
                    )
                after = self._access_for(user, workspace, direct_role, department_granted)
                if (
                    before["can_read"] != after["can_read"]
                    or before["can_manage"] != after["can_manage"]
                    or before["role"] != after["role"]
                ):
                    changes.append(
                        {
                            "workspace_id": workspace.id,
                            "workspace_name": workspace.name,
                            "workspace_status": workspace.status,
                            "before": {
                                "role": before["role"],
                                "can_read": before["can_read"],
                                "can_manage": before["can_manage"],
                            },
                            "after": {
                                "role": after["role"],
                                "can_read": after["can_read"],
                                "can_manage": after["can_manage"],
                            },
                        }
                    )
            return {
                "user_id": user.id,
                "username": user.username,
                "display_name": user.display_name,
                "current_department_id": user.department_id,
                "current_department_name": current_department.name if current_department else None,
                "new_department_id": next_department.id if next_department else None,
                "new_department_name": next_department.name if next_department else None,
                "changes": changes,
                "direct_permissions_preserved": True,
            }
