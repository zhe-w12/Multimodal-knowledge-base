"""Add a durable marker that prevents restoring a workspace during cleanup."""

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection


def upgrade(connection: Connection) -> None:
    if not inspect(connection).has_table("auth_workspaces"):
        return
    columns = {column["name"] for column in inspect(connection).get_columns("auth_workspaces")}
    if "deletion_in_progress" not in columns:
        connection.execute(
            text(
                "ALTER TABLE auth_workspaces "
                "ADD COLUMN deletion_in_progress BOOLEAN NOT NULL DEFAULT 0"
            )
        )
