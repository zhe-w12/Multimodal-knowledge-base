from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Department(Base):
    __tablename__ = "auth_departments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class User(Base):
    __tablename__ = "auth_users"
    __table_args__ = (
        CheckConstraint("platform_role IN ('user', 'platform_admin')", name="ck_auth_users_platform_role"),
        CheckConstraint("status IN ('active', 'suspended')", name="ck_auth_users_status"),
        Index("ix_auth_users_department", "department_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    platform_role: Mapped[str] = mapped_column(String(24), nullable=False, default="user")
    department_id: Mapped[Optional[str]] = mapped_column(ForeignKey("auth_departments.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    permissions_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    __table_args__ = (
        Index("ix_auth_sessions_user_active", "user_id", "revoked_at"),
        Index("ix_auth_sessions_expires", "absolute_expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("auth_users.id", ondelete="CASCADE"), nullable=False)
    token_digest: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    csrf_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    absolute_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class RegistrationSettings(Base):
    __tablename__ = "registration_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invite_digest: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    registration_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_by: Mapped[Optional[str]] = mapped_column(ForeignKey("auth_users.id", ondelete="SET NULL"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Workspace(Base):
    __tablename__ = "auth_workspaces"
    __table_args__ = (
        CheckConstraint("workspace_type IN ('general', 'authorized')", name="ck_auth_workspaces_type"),
        CheckConstraint("status IN ('active', 'archived')", name="ck_auth_workspaces_status"),
        CheckConstraint(
            "(workspace_type = 'general' AND general_marker = 'general' AND owner_user_id IS NULL) "
            "OR (workspace_type = 'authorized' AND general_marker IS NULL AND owner_user_id IS NOT NULL)",
            name="ck_auth_workspaces_general_marker",
        ),
        UniqueConstraint("general_marker", name="uq_auth_workspaces_general_marker"),
        Index("ix_auth_workspaces_owner", "owner_user_id"),
        Index("ix_auth_workspaces_type_status", "workspace_type", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[str] = mapped_column(String(1500), nullable=False, default="")
    workspace_type: Mapped[str] = mapped_column(String(16), nullable=False)
    general_marker: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("auth_users.id", ondelete="SET NULL"), nullable=True)
    owner_user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("auth_users.id", ondelete="SET NULL"), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class WorkspaceMembership(Base):
    __tablename__ = "auth_workspace_memberships"
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'reader', 'blocked')", name="ck_auth_workspace_membership_role"),
        UniqueConstraint("workspace_id", "user_id", name="uq_auth_workspace_membership_user"),
        Index("ix_auth_workspace_memberships_user", "user_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("auth_workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[str] = mapped_column(ForeignKey("auth_users.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    granted_by: Mapped[str] = mapped_column(ForeignKey("auth_users.id", ondelete="RESTRICT"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class DepartmentWorkspaceGrant(Base):
    __tablename__ = "auth_department_workspace_grants"
    __table_args__ = (
        UniqueConstraint("department_id", "workspace_id", name="uq_auth_department_workspace_grant"),
        Index("ix_auth_department_workspace_grants_workspace", "workspace_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    department_id: Mapped[str] = mapped_column(ForeignKey("auth_departments.id", ondelete="CASCADE"), nullable=False)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("auth_workspaces.id", ondelete="CASCADE"), nullable=False)
    granted_by: Mapped[str] = mapped_column(ForeignKey("auth_users.id", ondelete="RESTRICT"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class LoginAttempt(Base):
    __tablename__ = "auth_login_attempts"

    bucket_digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(24), nullable=False, default="login")
    account_digest: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    ip_digest: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    window_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class AuditLog(Base):
    __tablename__ = "auth_audit_logs"
    __table_args__ = (Index("ix_auth_audit_created_at", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("auth_users.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    target_type: Mapped[str] = mapped_column(String(40), nullable=False)
    target_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    details: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
