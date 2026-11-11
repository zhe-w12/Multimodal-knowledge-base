from __future__ import annotations

import base64
import hashlib
import hmac
import re
import unicodedata

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,31}$")
RESERVED_USERNAMES = frozenset(
    {
        "admin",
        "administrator",
        "root",
        "system",
        "support",
        "owner",
        "api",
        "null",
    }
)
COMMON_PASSWORDS = frozenset(
    {
        "123456789012345",
        "1234567890123456",
        "password1234567",
        "password12345678",
        "qwerty123456789",
        "letmein12345678",
        "iloveyou1234567",
        "admin1234567890",
        "changeme1234567",
        "welcome12345678",
    }
)

PASSWORD_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=1,
    hash_len=32,
    salt_len=16,
)
DUMMY_PASSWORD = "not-a-real-user-credential-9Qz!"
DUMMY_PASSWORD_HASH = PASSWORD_HASHER.hash(DUMMY_PASSWORD)


def normalize_username(value: str) -> str:
    username = str(value or "").strip()
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError("账号须为 3～32 位字母、数字、下划线或短横线，且以字母或数字开头")
    return username.lower()


def normalize_login_username(value: str) -> str | None:
    try:
        return normalize_username(value)
    except ValueError:
        return None


def normalize_password(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("密码格式无效")
    password = unicodedata.normalize("NFC", value)
    if not 8 <= len(password) <= 16:
        raise ValueError("密码长度须为 8～16 个字符")
    if any(not character.isprintable() for character in password):
        raise ValueError("密码不能包含换行、制表符或不可见控制字符")
    return password


def validate_new_password(
    value: str,
    username: str,
    *,
    allow_reserved: bool = False,
    allow_username_in_password: bool = False,
) -> str:
    password = normalize_password(value)
    compact_password = "".join(
        character for character in unicodedata.normalize("NFKC", password).casefold() if character.isalnum()
    )
    if compact_password in COMMON_PASSWORDS:
        raise ValueError("该密码过于常见，请换一个更难猜的密码")
    # Account text in a password is allowed; minimum length and common-password
    # checks still apply. Keep the compatibility argument for existing callers.
    if not allow_reserved and username.lower() in RESERVED_USERNAMES:
        raise ValueError("该账号名称不可注册")
    return password


def hash_password(password: str) -> str:
    return PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        normalized = normalize_password(password)
    except ValueError:
        return False
    try:
        return PASSWORD_HASHER.verify(password_hash, normalized)
    except (InvalidHashError, VerificationError, VerifyMismatchError, ValueError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    try:
        return PASSWORD_HASHER.check_needs_rehash(password_hash)
    except (InvalidHashError, VerificationError, ValueError):
        return False


def csrf_token_for_session(secret: bytes, raw_session_token: str) -> str:
    digest = hmac.new(secret, b"csrf:" + raw_session_token.encode("ascii"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
