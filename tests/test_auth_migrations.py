from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from manufacturing_rag.auth.migrations.v0004_workspace_deletion_state import upgrade


def test_workspace_deletion_migration_adds_default_false_column():
    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE auth_workspaces (id VARCHAR(36) PRIMARY KEY)"))
            upgrade(connection)

            columns = {column["name"]: column for column in inspect(connection).get_columns("auth_workspaces")}
            assert "deletion_in_progress" in columns

            connection.execute(text("INSERT INTO auth_workspaces (id) VALUES ('existing-workspace')"))
            deletion_state = connection.scalar(
                text("SELECT deletion_in_progress FROM auth_workspaces WHERE id = 'existing-workspace'")
            )
            assert deletion_state in (False, 0)
    finally:
        engine.dispose()
