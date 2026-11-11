from __future__ import annotations

import argparse
import getpass
import sys

from manufacturing_rag.auth.store import AccountAlreadyExists, AuthServiceUnavailable, AuthStore


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m manufacturing_rag.auth.admin")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create-platform-admin", help="交互式创建首个平台管理员账号")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.command != "create-platform-admin":
        return 2

    username = input("平台管理员账号：").strip()
    password = getpass.getpass("设置密码（8～16 个字符）：")
    confirmation = getpass.getpass("再次输入密码：")
    if password != confirmation:
        print("两次输入的密码不一致。", file=sys.stderr)
        return 2

    try:
        user = AuthStore.from_environment().create_initial_platform_admin(username, password)
    except (ValueError, AccountAlreadyExists, RuntimeError, AuthServiceUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"已创建首个平台管理员：{user['username']}（{user['id']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
