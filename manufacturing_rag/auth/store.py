from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, and_, create_engine, delete, event, func, insert, select
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import aliased, sessionmaker
from sqlalchemy.pool import StaticPool

from manufacturing_rag.auth.migrations import MIGRATIONS
from manufacturing_rag.auth.models import (
    AuditLog,
    AuthSession,
    Base,
    Department,
    LoginAttempt,
    RegistrationSettings,
    User,
    utcnow,
)
from manufacturing_rag.auth.security import (
    DUMMY_PASSWORD,
    DUMMY_PASSWORD_HASH,
    PASSWORD_HASHER,
    hash_password,
    normalize_invite_code,
    normalize_login_username,
    normalize_password,
    normalize_username,
    password_needs_rehash,
    validate_new_password,
    verify_password,
    csrf_token_for_session,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MIGRATION_TABLE = Table(
    "auth_schema_migrations",
    MetaData(),
    Column("version", Integer, primary_key=True),
    Column("description", String(255), nullable=False),
    Column("applied_at", DateTime(timezone=True), nullable=False),
)


class AuthServiceUnavailable(RuntimeError):
    pass


class InvalidCredentials(RuntimeError):
    pass


class InvalidInvitation(RuntimeError):
    pass


class RegistrationClosed(RuntimeError):
    pass


class AccountAlreadyExists(RuntimeError):
    pass


class LoginTemporarilyLimited(RuntimeError):
    pass


class CurrentPasswordIncorrect(RuntimeError):
    pass


class NewPasswordUnchanged(RuntimeError):
    pass


class AccountUnavailable(RuntimeError):
    pass


class PlatformAdminRequired(RuntimeError):
    pass


class LastPlatformAdminError(RuntimeError):
    pass


@dataclass(frozen=True)
class Principal:
    user: dict[str, Any]
    session_id: str
    token_digest: str
    csrf_token: str


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _invite_pepper(data_dir: Path, provided: bytes | None = None) -> bytes:
    if provided is not None:
        if len(provided) < 32:
            raise ValueError("invite pepper must contain at least 32 bytes")
        return provided

    configured = os.getenv("AUTH_INVITE_PEPPER", "")
    if configured:
        secret = configured.encode("utf-8")
        if len(secret) < 32:
            raise ValueError("AUTH_INVITE_PEPPER must contain at least 32 bytes")
        return secret

    key_path = _project_path(os.getenv("AUTH_INVITE_PEPPER_FILE", str(data_dir / "invite.key")))
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        secret = key_path.read_bytes()
    else:
        secret = secrets.token_bytes(32)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(secret)
            handle.flush()
            os.fsync(handle.fileno())
    if len(secret) < 32:
        raise ValueError("invite pepper file must contain at least 32 bytes")
    if os.name == "posix":
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
    return secret


def _database_url(data_dir: Path) -> tuple[str | URL, dict[str, Any], Any]:
    configured = os.getenv("AUTH_DATABASE_URL", "").strip()
    if not configured:
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        configured_url: str | URL = URL.create("sqlite", database=str(data_dir / "auth.sqlite3"))
    else:
        configured_url = make_url(configured)

    parsed = make_url(configured_url)
    connect_args: dict[str, Any] = {}
    pool_class = None
    if parsed.drivername.startswith("sqlite"):
        database = parsed.database
        if not database or database == ":memory:":
            connect_args["check_same_thread"] = False
            pool_class = StaticPool
        else:
            db_path = _project_path(database)
            db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            configured_url = parsed.set(database=str(db_path))
            connect_args["check_same_thread"] = False
    return configured_url, connect_args, pool_class


class AuthStore:
    """Persistent identity store. The database is never replaced with in-memory users."""

    def __init__(self, database_url: str | URL | None = None, invite_pepper: bytes | None = None):
        data_dir = _project_path(os.getenv("AUTH_DATA_DIR", "data/auth"))
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.invite_pepper = _invite_pepper(data_dir, invite_pepper)
        if database_url is None:
            database_url, connect_args, pool_class = _database_url(data_dir)
        else:
            database_url = make_url(database_url) if isinstance(database_url, str) else database_url
            connect_args = {}
            pool_class = None
            if database_url.drivername.startswith("sqlite"):
                if not database_url.database or database_url.database == ":memory:":
                    connect_args["check_same_thread"] = False
                    pool_class = StaticPool
                else:
                    db_path = _project_path(database_url.database)
                    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    database_url = database_url.set(database=str(db_path))

        engine_options: dict[str, Any] = {"pool_pre_ping": True}
        if connect_args:
            engine_options["connect_args"] = connect_args
        if pool_class is not None:
            engine_options["poolclass"] = pool_class
        self.engine = create_engine(database_url, **engine_options)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._migration_lock = threading.RLock()
        self._registration_rate_lock = threading.RLock()
        self._migrated = False

        if self.engine.dialect.name == "sqlite":
            @event.listens_for(self.engine, "connect")
            def configure_sqlite(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.close()

    @classmethod
    def from_environment(cls) -> "AuthStore":
        return cls()

    def ensure_schema(self) -> None:
        if self._migrated:
            return
        with self._migration_lock:
            if self._migrated:
                return
            try:
                with self.engine.begin() as connection:
                    MIGRATION_TABLE.create(bind=connection, checkfirst=True)
                for version, description, upgrade in MIGRATIONS:
                    with self.engine.begin() as connection:
                        applied = connection.execute(
                            select(MIGRATION_TABLE.c.version).where(MIGRATION_TABLE.c.version == version)
                        ).first()
                        if applied:
                            continue
                        upgrade(connection)
                        connection.execute(
                            insert(MIGRATION_TABLE).values(
                                version=version,
                                description=description,
                                applied_at=utcnow(),
                            )
                        )
                self._migrated = True
            except SQLAlchemyError as exc:
                raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    @staticmethod
    def public_user(user: User) -> dict[str, Any]:
        return {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "platform_role": user.platform_role,
            "department_id": user.department_id,
            "status": user.status,
            "permissions_version": user.permissions_version,
        }

    def _invite_digest(self, code: str) -> str:
        return hmac.new(self.invite_pepper, b"invite:" + code.encode("ascii"), hashlib.sha256).hexdigest()

    def _csrf_token(self, raw_session_token: str) -> str:
        return csrf_token_for_session(self.invite_pepper, raw_session_token)

    @staticmethod
    def _token_digest(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("ascii")).hexdigest()

    def _new_session_values(self) -> tuple[str, str, AuthSession]:
        raw_token = secrets.token_urlsafe(32)
        csrf_token = self._csrf_token(raw_token)
        now = utcnow()
        absolute_hours = _positive_int("AUTH_SESSION_ABSOLUTE_HOURS", 12)
        session = AuthSession(
            id=str(uuid4()),
            user_id="",
            token_digest=self._token_digest(raw_token),
            csrf_digest=hashlib.sha256(csrf_token.encode("ascii")).hexdigest(),
            created_at=now,
            last_seen_at=now,
            absolute_expires_at=now + timedelta(hours=absolute_hours),
        )
        return raw_token, csrf_token, session

    def _audit(self, db, actor_user_id: str | None, action: str, target_type: str, target_id: str | None, details=None) -> None:
        db.add(
            AuditLog(
                id=str(uuid4()),
                actor_user_id=actor_user_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                details=json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                created_at=utcnow(),
            )
        )

    @staticmethod
    def serialize_audit_entry(log: AuditLog, actor_username: str | None, target_username: str | None = None) -> dict[str, Any]:
        try:
            details = json.loads(log.details or "{}")
        except (TypeError, json.JSONDecodeError):
            details = {}
        if not isinstance(details, dict):
            details = {}
        visible_target_id = log.target_id if log.target_type in {"workspace", "department", "registration"} else None
        return {
            "id": log.id,
            "action": log.action,
            "actor_username": actor_username or "服务器初始化",
            "target_type": log.target_type,
            "target_id": visible_target_id,
            "target_username": target_username,
            "details": details,
            "created_at": log.created_at.isoformat(),
        }

    def create_initial_platform_admin(self, username: str, password: str) -> dict[str, Any]:
        try:
            normalized_username = normalize_username(username)
            # The server-only bootstrap flow may intentionally use the operator's chosen account phrase.
            normalized_password = validate_new_password(
                password,
                normalized_username,
                allow_reserved=True,
                allow_username_in_password=True,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        password_hash = hash_password(normalized_password)
        user = User(
            id=str(uuid4()),
            username=normalized_username,
            display_name=normalized_username,
            password_hash=password_hash,
            platform_role="platform_admin",
            status="active",
            permissions_version=1,
        )
        self.ensure_schema()
        try:
            with self.Session.begin() as db:
                settings = db.get(RegistrationSettings, 1, with_for_update=True)
                if not settings:
                    db.add(RegistrationSettings(id=1, registration_enabled=False, version=0))
                    db.flush()
                active_admin = db.scalar(
                    select(User.id)
                    .where(User.platform_role == "platform_admin", User.status == "active")
                    .with_for_update()
                    .limit(1)
                )
                if active_admin:
                    raise RuntimeError("已有启用的平台管理员；不能再次执行首次初始化")
                db.add(user)
                db.flush()
                self._audit(db, None, "platform_admin_bootstrapped", "user", user.id)
            return self.public_user(user)
        except IntegrityError as exc:
            raise AccountAlreadyExists("该账号已被使用") from exc
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def _attempt_buckets(self, username_key: str, ip_address: str) -> tuple[tuple[str, int, str, str | None, str | None], ...]:
        account_digest = hmac.new(
            self.invite_pepper,
            f"rate-account:{username_key}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        ip_digest = hmac.new(
            self.invite_pepper,
            f"rate-ip:{ip_address}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        scopes = (
            ("account_ip", f"{username_key}\0{ip_address}", _positive_int("AUTH_LOGIN_PAIR_LIMIT", 5)),
            ("account", username_key, _positive_int("AUTH_LOGIN_ACCOUNT_LIMIT", 20)),
            ("ip", ip_address, _positive_int("AUTH_LOGIN_IP_LIMIT", 60)),
        )
        return tuple(
            (
                hmac.new(self.invite_pepper, f"login:{scope}:{value}".encode("utf-8"), hashlib.sha256).hexdigest(),
                limit,
                scope,
                account_digest if scope != "ip" else None,
                ip_digest if scope != "account" else None,
            )
            for scope, value, limit in scopes
        )

    def _check_rate_limit(self, buckets: tuple[tuple[str, int, str, str | None, str | None], ...]) -> None:
        self.ensure_schema()
        now = utcnow()
        try:
            with self.Session() as db:
                for digest, _limit, _scope, _account_digest, _ip_digest in buckets:
                    attempt = db.get(LoginAttempt, digest)
                    if attempt and attempt.locked_until and _as_utc(attempt.locked_until) > now:
                        raise LoginTemporarilyLimited("登录暂时受限")
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def _record_failed_login(self, buckets: tuple[tuple[str, int, str, str | None, str | None], ...]) -> None:
        now = utcnow()
        window_minutes = _positive_int("AUTH_LOGIN_WINDOW_MINUTES", 15)
        lock_minutes = _positive_int("AUTH_LOGIN_LOCK_MINUTES", 15)
        try:
            with self.Session.begin() as db:
                for digest, limit, scope, account_digest, ip_digest in buckets:
                    attempt = db.get(LoginAttempt, digest, with_for_update=True)
                    if attempt is None:
                        attempt = LoginAttempt(
                            bucket_digest=digest,
                            scope_type=scope,
                            account_digest=account_digest,
                            ip_digest=ip_digest,
                            failure_count=0,
                            window_started_at=now,
                        )
                        db.add(attempt)
                    elif now - _as_utc(attempt.window_started_at) >= timedelta(minutes=window_minutes):
                        attempt.failure_count = 0
                        attempt.window_started_at = now
                        attempt.locked_until = None
                    attempt.failure_count += 1
                    if attempt.failure_count >= limit:
                        attempt.locked_until = now + timedelta(minutes=lock_minutes)
                    attempt.updated_at = now
        except IntegrityError:
            # A concurrent first failure may create the same bucket; retry once against the now-existing rows.
            try:
                with self.Session.begin() as db:
                    for digest, limit, scope, account_digest, ip_digest in buckets:
                        attempt = db.get(LoginAttempt, digest, with_for_update=True)
                        if attempt is None:
                            db.add(
                                LoginAttempt(
                                    bucket_digest=digest,
                                    scope_type=scope,
                                    account_digest=account_digest,
                                    ip_digest=ip_digest,
                                    failure_count=1,
                                    window_started_at=now,
                                )
                            )
                        else:
                            attempt.failure_count += 1
                            if attempt.failure_count >= limit:
                                attempt.locked_until = now + timedelta(minutes=lock_minutes)
                            attempt.updated_at = now
            except SQLAlchemyError as exc:
                raise AuthServiceUnavailable("账号数据服务暂不可用") from exc
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def _clear_successful_login_limits(self, buckets: tuple[tuple[str, int], ...]) -> None:
        account_buckets = [bucket[0] for bucket in buckets[:2]]
        try:
            with self.Session.begin() as db:
                db.execute(delete(LoginAttempt).where(LoginAttempt.bucket_digest.in_(account_buckets)))
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def check_registration_rate_limit(self, ip_address: str) -> None:
        buckets = self._attempt_buckets("registration-invitation", str(ip_address or "unknown")[:128])
        self._check_rate_limit(buckets)

    def record_registration_failure(self, ip_address: str) -> None:
        buckets = self._attempt_buckets("registration-invitation", str(ip_address or "unknown")[:128])
        self._record_failed_login(buckets)

    def _consume_registration_slot(self, ip_address: str) -> None:
        digest = hmac.new(
            self.invite_pepper,
            f"registration-success:{str(ip_address or 'unknown')[:128]}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        limit = _positive_int("AUTH_REGISTRATION_IP_LIMIT", 10)
        window_minutes = _positive_int("AUTH_REGISTRATION_WINDOW_MINUTES", 60)
        lock_minutes = _positive_int("AUTH_REGISTRATION_LOCK_MINUTES", 60)
        now = utcnow()
        self.ensure_schema()

        def update_slot() -> None:
            with self.Session.begin() as db:
                attempt = db.get(LoginAttempt, digest, with_for_update=True)
                if attempt and attempt.locked_until and _as_utc(attempt.locked_until) > now:
                    raise LoginTemporarilyLimited("注册请求过于频繁")
                if attempt is None:
                    attempt = LoginAttempt(
                        bucket_digest=digest,
                        scope_type="registration_ip",
                        ip_digest=hmac.new(
                            self.invite_pepper,
                            f"rate-ip:{str(ip_address or 'unknown')[:128]}".encode("utf-8"),
                            hashlib.sha256,
                        ).hexdigest(),
                        failure_count=0,
                        window_started_at=now,
                    )
                    db.add(attempt)
                elif now - _as_utc(attempt.window_started_at) >= timedelta(minutes=window_minutes):
                    attempt.failure_count = 0
                    attempt.window_started_at = now
                    attempt.locked_until = None
                if attempt.failure_count >= limit:
                    attempt.locked_until = now + timedelta(minutes=lock_minutes)
                    raise LoginTemporarilyLimited("注册请求过于频繁")
                attempt.failure_count += 1
                attempt.updated_at = now
                if attempt.failure_count >= limit:
                    attempt.locked_until = now + timedelta(minutes=lock_minutes)

        with self._registration_rate_lock:
            try:
                update_slot()
            except IntegrityError:
                try:
                    update_slot()
                except SQLAlchemyError as exc:
                    raise AuthServiceUnavailable("账号数据服务暂不可用") from exc
            except LoginTemporarilyLimited:
                raise
            except SQLAlchemyError as exc:
                raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def login(self, username: str, password: str, ip_address: str) -> tuple[dict[str, Any], str, str]:
        normalized_username = normalize_login_username(username)
        username_key = normalized_username or str(username or "").strip().lower()[:64]
        safe_ip = str(ip_address or "unknown")[:128]
        buckets = self._attempt_buckets(username_key, safe_ip)
        self._check_rate_limit(buckets)

        self.ensure_schema()
        try:
            with self.Session() as db:
                user = db.scalar(select(User).where(User.username == normalized_username)) if normalized_username else None
                user_id = user.id if user else None
                stored_hash = user.password_hash if user else DUMMY_PASSWORD_HASH
                account_active = bool(user and user.status == "active")
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

        try:
            submitted_password = normalize_password(password)
            password_format_valid = True
        except ValueError:
            submitted_password = DUMMY_PASSWORD
            password_format_valid = False
        password_matches = verify_password(stored_hash, submitted_password) if password_format_valid and user_id else False
        if not user_id or not account_active or not password_matches:
            if not password_format_valid or not user_id:
                try:
                    PASSWORD_HASHER.verify(DUMMY_PASSWORD_HASH, DUMMY_PASSWORD)
                except Exception:
                    pass
            self._record_failed_login(buckets)
            raise InvalidCredentials("账号或密码错误")

        original_hash = stored_hash
        rehashed_password = hash_password(submitted_password) if password_needs_rehash(original_hash) else None

        raw_token, csrf_token, auth_session = self._new_session_values()
        try:
            with self.Session.begin() as db:
                user = db.get(User, user_id, with_for_update=True)
                if not user or user.status != "active" or user.password_hash != original_hash:
                    raise InvalidCredentials("账号或密码错误")
                if rehashed_password:
                    user.password_hash = rehashed_password
                auth_session.user_id = user.id
                db.add(auth_session)
                self._audit(db, user.id, "login_succeeded", "session", auth_session.id)
                public_user = self.public_user(user)
            self._clear_successful_login_limits(buckets)
            return public_user, raw_token, csrf_token
        except InvalidCredentials:
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def register(
        self,
        username: str,
        display_name: str | None,
        password: str,
        invite_code: str,
        ip_address: str = "unknown",
    ) -> tuple[dict[str, Any], str, str]:
        normalized_username = normalize_username(username)
        normalized_password = validate_new_password(password, normalized_username)
        normalized_display_name = str(display_name or "").strip()
        if normalized_display_name and not 2 <= len(normalized_display_name) <= 30:
            raise ValueError("显示名称须为 2～30 个字符")
        if any(not character.isprintable() for character in normalized_display_name):
            raise ValueError("显示名称不能包含控制字符")
        if not normalized_display_name:
            normalized_display_name = normalized_username

        normalized_code = normalize_invite_code(invite_code)
        if normalized_code is None:
            raise InvalidInvitation("邀请码无效或注册暂未开放")
        invite_digest = self._invite_digest(normalized_code)
        password_hash = hash_password(normalized_password)
        self._consume_registration_slot(ip_address)
        user = User(
            id=str(uuid4()),
            username=normalized_username,
            display_name=normalized_display_name,
            password_hash=password_hash,
            platform_role="user",
            status="active",
            permissions_version=1,
        )
        raw_token, csrf_token, auth_session = self._new_session_values()
        self.ensure_schema()
        try:
            with self.Session.begin() as db:
                settings = db.get(RegistrationSettings, 1, with_for_update=True)
                if not settings or not settings.registration_enabled:
                    raise RegistrationClosed("邀请码无效或注册暂未开放")
                if not settings.invite_digest or not hmac.compare_digest(settings.invite_digest, invite_digest):
                    raise InvalidInvitation("邀请码无效或注册暂未开放")
                db.add(user)
                db.flush()
                auth_session.user_id = user.id
                db.add(auth_session)
                self._audit(db, user.id, "account_registered", "user", user.id)
                public_user = self.public_user(user)
            return public_user, raw_token, csrf_token
        except (InvalidInvitation, RegistrationClosed):
            raise
        except IntegrityError as exc:
            raise AccountAlreadyExists("该账号已被使用") from exc
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def current_principal(self, raw_token: str | None) -> Principal | None:
        if not raw_token or len(raw_token) > 256:
            return None
        self.ensure_schema()
        digest = self._token_digest(raw_token)
        now = utcnow()
        idle_hours = _positive_int("AUTH_SESSION_IDLE_HOURS", 2)
        try:
            with self.Session.begin() as db:
                auth_session = db.scalar(
                    select(AuthSession).where(AuthSession.token_digest == digest).with_for_update()
                )
                if not auth_session:
                    return None
                user = db.get(User, auth_session.user_id)
                idle_expired = now - _as_utc(auth_session.last_seen_at) >= timedelta(hours=idle_hours)
                absolute_expired = now >= _as_utc(auth_session.absolute_expires_at)
                if auth_session.revoked_at or not user or user.status != "active" or idle_expired or absolute_expired:
                    if not auth_session.revoked_at:
                        auth_session.revoked_at = now
                    return None
                csrf_token = self._csrf_token(raw_token)
                csrf_digest = hashlib.sha256(csrf_token.encode("ascii")).hexdigest()
                if not hmac.compare_digest(auth_session.csrf_digest, csrf_digest):
                    auth_session.revoked_at = now
                    return None
                auth_session.last_seen_at = now
                public_user = self.public_user(user)
                department = db.get(Department, user.department_id) if user.department_id else None
                public_user["department_name"] = department.name if department else None
                return Principal(
                    user=public_user,
                    session_id=auth_session.id,
                    token_digest=digest,
                    csrf_token=csrf_token,
                )
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def revoke_session(self, principal: Principal) -> None:
        self.ensure_schema()
        now = utcnow()
        try:
            with self.Session.begin() as db:
                auth_session = db.get(AuthSession, principal.session_id, with_for_update=True)
                if auth_session and not auth_session.revoked_at:
                    auth_session.revoked_at = now
                    self._audit(db, principal.user["id"], "logout", "session", auth_session.id)
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def revoke_all_sessions(self, user_id: str, actor_user_id: str, action: str = "logout_all") -> None:
        self.ensure_schema()
        now = utcnow()
        try:
            with self.Session.begin() as db:
                db.execute(
                    AuthSession.__table__.update()
                    .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
                    .values(revoked_at=now)
                )
                self._audit(db, actor_user_id, action, "user", user_id)
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def change_password(self, user_id: str, current_password: str, new_password: str) -> None:
        self.ensure_schema()
        try:
            with self.Session() as db:
                user = db.get(User, user_id)
                if not user or user.status != "active":
                    raise AccountUnavailable("账号当前不可用")
                current_hash = user.password_hash
                username = user.username
            if not verify_password(current_hash, current_password):
                raise CurrentPasswordIncorrect("当前密码不正确")
            normalized_new_password = validate_new_password(new_password, username)
            if verify_password(current_hash, normalized_new_password):
                raise NewPasswordUnchanged("新密码不能与当前密码相同")
            new_hash = hash_password(normalized_new_password)
            now = utcnow()
            with self.Session.begin() as db:
                user = db.get(User, user_id, with_for_update=True)
                if not user or user.status != "active" or user.password_hash != current_hash:
                    raise CurrentPasswordIncorrect("当前密码不正确")
                user.password_hash = new_hash
                user.permissions_version += 1
                user.updated_at = now
                db.execute(
                    AuthSession.__table__.update()
                    .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
                    .values(revoked_at=now)
                )
                self._audit(db, user_id, "password_changed", "user", user_id)
        except (AccountUnavailable, CurrentPasswordIncorrect, NewPasswordUnchanged, ValueError):
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def _require_platform_admin(self, db, user_id: str) -> User:
        user = db.get(User, user_id, with_for_update=True)
        if not user or user.status != "active" or user.platform_role != "platform_admin":
            raise PlatformAdminRequired("需要平台管理员权限")
        return user

    def registration_status(self) -> dict[str, Any]:
        self.ensure_schema()
        try:
            with self.Session() as db:
                settings = db.get(RegistrationSettings, 1)
                return {
                    "configured": bool(settings and settings.invite_digest),
                    "registration_enabled": bool(settings and settings.registration_enabled),
                }
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def record_audit_event(
        self,
        actor_user_id: str,
        action: str,
        target_type: str,
        target_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_schema()
        try:
            with self.Session.begin() as db:
                actor = db.get(User, actor_user_id)
                if actor is None:
                    return
                self._audit(db, actor.id, action, target_type, target_id, details)
        except AccountUnavailable:
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号审计数据暂不可用") from exc

    def list_audit_logs(self, actor_user_id: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        bounded_limit = min(max(int(limit), 1), 100)
        bounded_offset = max(int(offset), 0)
        self.ensure_schema()
        actor_alias = aliased(User)
        target_alias = aliased(User)
        try:
            with self.Session() as db:
                actor = db.get(User, actor_user_id)
                if not actor or actor.status != "active" or actor.platform_role != "platform_admin":
                    raise PlatformAdminRequired("需要平台管理员权限")
                total = int(db.scalar(select(func.count(AuditLog.id))) or 0)
                rows = db.execute(
                    select(AuditLog, actor_alias.username, target_alias.username)
                    .outerjoin(actor_alias, actor_alias.id == AuditLog.actor_user_id)
                    .outerjoin(
                        target_alias,
                        and_(AuditLog.target_type == "user", target_alias.id == AuditLog.target_id),
                    )
                    .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
                    .limit(bounded_limit)
                    .offset(bounded_offset)
                ).all()
                entries = [
                    self.serialize_audit_entry(log, actor_name, target_name)
                    for log, actor_name, target_name in rows
                ]
                return {"entries": entries, "total": total, "limit": bounded_limit, "offset": bounded_offset}
        except PlatformAdminRequired:
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号审计数据暂不可用") from exc

    def list_users(self, actor_user_id: str, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        self.ensure_schema()
        term = query.strip().casefold()
        bounded_limit = min(max(int(limit), 1), 100)
        try:
            with self.Session() as db:
                actor = db.get(User, actor_user_id)
                if not actor or actor.status != "active" or actor.platform_role != "platform_admin":
                    raise PlatformAdminRequired("需要平台管理员权限")
                statement = select(User).order_by(User.username).limit(bounded_limit)
                if term:
                    escaped_term = term.replace("%", "\\%").replace("_", "\\_")
                    pattern = f"%{escaped_term}%"
                    statement = statement.where(
                        User.username.ilike(pattern, escape="\\")
                        | User.display_name.ilike(pattern, escape="\\")
                    )
                return [self.public_user(user) for user in db.scalars(statement).all()]
        except PlatformAdminRequired:
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def set_platform_role(self, actor_user_id: str, target_user_id: str, role: str) -> dict[str, Any]:
        if role not in {"user", "platform_admin"}:
            raise ValueError("平台角色无效")
        self.ensure_schema()
        try:
            with self.Session.begin() as db:
                actor = self._require_platform_admin(db, actor_user_id)
                target = db.get(User, target_user_id, with_for_update=True)
                if target is None:
                    raise ValueError("用户不存在")
                if role == "platform_admin" and target.status != "active":
                    raise ValueError("只能将启用账号设为平台管理员")
                removing_active_admin = target.status == "active" and target.platform_role == "platform_admin" and role != "platform_admin"
                if removing_active_admin:
                    active_admin_ids = db.scalars(
                        select(User.id)
                        .where(User.status == "active", User.platform_role == "platform_admin")
                        .with_for_update()
                    ).all()
                    if len(active_admin_ids) <= 1:
                        raise LastPlatformAdminError("至少保留一名启用的平台管理员")
                old_role = target.platform_role
                target.platform_role = role
                target.permissions_version += 1
                if old_role != role:
                    db.execute(
                        AuthSession.__table__.update()
                        .where(AuthSession.user_id == target.id, AuthSession.revoked_at.is_(None))
                        .values(revoked_at=utcnow())
                    )
                self._audit(
                    db,
                    actor.id,
                    "platform_role_changed",
                    "user",
                    target.id,
                    {"old_role": old_role, "new_role": role},
                )
                return self.public_user(target)
        except (PlatformAdminRequired, LastPlatformAdminError, ValueError):
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def set_user_status(self, actor_user_id: str, target_user_id: str, status: str) -> dict[str, Any]:
        if status not in {"active", "suspended"}:
            raise ValueError("账号状态无效")
        self.ensure_schema()
        try:
            with self.Session.begin() as db:
                actor = self._require_platform_admin(db, actor_user_id)
                target = db.get(User, target_user_id, with_for_update=True)
                if target is None:
                    raise ValueError("用户不存在")
                removing_active_admin = target.status == "active" and target.platform_role == "platform_admin" and status != "active"
                if removing_active_admin:
                    active_admin_ids = db.scalars(
                        select(User.id)
                        .where(User.status == "active", User.platform_role == "platform_admin")
                        .with_for_update()
                    ).all()
                    if len(active_admin_ids) <= 1:
                        raise LastPlatformAdminError("至少保留一名启用的平台管理员")
                old_status = target.status
                target.status = status
                target.permissions_version += 1
                if status != "active":
                    db.execute(
                        AuthSession.__table__.update()
                        .where(AuthSession.user_id == target.id, AuthSession.revoked_at.is_(None))
                        .values(revoked_at=utcnow())
                    )
                self._audit(
                    db,
                    actor.id,
                    "user_status_changed",
                    "user",
                    target.id,
                    {"old_status": old_status, "new_status": status},
                )
                return self.public_user(target)
        except (PlatformAdminRequired, LastPlatformAdminError, ValueError):
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def invitation_status(self, actor_user_id: str) -> dict[str, Any]:
        self.ensure_schema()
        try:
            with self.Session() as db:
                actor = db.get(User, actor_user_id)
                if not actor or actor.status != "active" or actor.platform_role != "platform_admin":
                    raise PlatformAdminRequired("需要平台管理员权限")
                settings = db.get(RegistrationSettings, 1)
                updated_by = db.get(User, settings.updated_by) if settings and settings.updated_by else None
                return {
                    "configured": bool(settings and settings.invite_digest),
                    "registration_enabled": bool(settings and settings.registration_enabled),
                    "version": settings.version if settings else 0,
                    "updated_at": settings.updated_at.isoformat() if settings and settings.updated_at else None,
                    "updated_by": updated_by.username if updated_by else None,
                }
        except PlatformAdminRequired:
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def reset_invitation(self, actor_user_id: str) -> tuple[str, dict[str, Any]]:
        self.ensure_schema()
        try:
            with self.Session.begin() as db:
                actor = self._require_platform_admin(db, actor_user_id)
                settings = db.get(RegistrationSettings, 1, with_for_update=True)
                if not settings:
                    settings = RegistrationSettings(id=1, registration_enabled=False, version=0)
                    db.add(settings)
                    db.flush()
                old_digest = settings.invite_digest
                code = ""
                digest = ""
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                for _ in range(20):
                    code = "".join(secrets.choice(alphabet) for _ in range(6))
                    digest = self._invite_digest(code)
                    if not old_digest or not hmac.compare_digest(old_digest, digest):
                        break
                else:
                    raise RuntimeError("无法生成与旧邀请码不同的新邀请码")
                settings.invite_digest = digest
                settings.updated_by = actor.id
                settings.updated_at = utcnow()
                settings.version += 1
                self._audit(db, actor.id, "registration_invitation_reset", "registration", "global", {"version": settings.version})
                status = {
                    "configured": True,
                    "registration_enabled": bool(settings.registration_enabled),
                    "version": settings.version,
                    "updated_at": settings.updated_at.isoformat(),
                    "updated_by": actor.username,
                }
            return code, status
        except PlatformAdminRequired:
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def set_registration_enabled(self, actor_user_id: str, enabled: bool) -> dict[str, Any]:
        self.ensure_schema()
        try:
            with self.Session.begin() as db:
                actor = self._require_platform_admin(db, actor_user_id)
                settings = db.get(RegistrationSettings, 1, with_for_update=True)
                if not settings:
                    settings = RegistrationSettings(id=1, registration_enabled=False, version=0)
                    db.add(settings)
                    db.flush()
                if enabled and not settings.invite_digest:
                    raise RegistrationClosed("请先生成邀请码，再开放注册")
                settings.registration_enabled = bool(enabled)
                settings.updated_by = actor.id
                settings.updated_at = utcnow()
                settings.version += 1
                self._audit(
                    db,
                    actor.id,
                    "registration_enabled" if enabled else "registration_disabled",
                    "registration",
                    "global",
                    {"version": settings.version},
                )
                updated_at = settings.updated_at.isoformat()
                version = settings.version
            return {
                "configured": True if settings.invite_digest else False,
                "registration_enabled": bool(enabled),
                "version": version,
                "updated_at": updated_at,
                "updated_by": actor.username,
            }
        except (PlatformAdminRequired, RegistrationClosed):
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc

    def clear_login_limits(self, actor_user_id: str, username: str) -> int:
        normalized_username = normalize_username(username)
        account_digest = hmac.new(
            self.invite_pepper,
            f"rate-account:{normalized_username}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        self.ensure_schema()
        try:
            with self.Session.begin() as db:
                actor = self._require_platform_admin(db, actor_user_id)
                result = db.execute(
                    delete(LoginAttempt).where(LoginAttempt.account_digest == account_digest)
                )
                removed_count = int(result.rowcount or 0)
                self._audit(
                    db,
                    actor.id,
                    "login_limits_cleared",
                    "account_rate_limit",
                    account_digest,
                    {"removed_bucket_count": removed_count},
                )
            return removed_count
        except (PlatformAdminRequired, ValueError):
            raise
        except SQLAlchemyError as exc:
            raise AuthServiceUnavailable("账号数据服务暂不可用") from exc
