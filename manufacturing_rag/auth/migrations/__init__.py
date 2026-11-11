"""Versioned schema migrations for persistent authentication data."""

from manufacturing_rag.auth.migrations.v0001_identity import upgrade as upgrade_v0001
from manufacturing_rag.auth.migrations.v0002_workspaces import upgrade as upgrade_v0002
from manufacturing_rag.auth.migrations.v0003_conversations import upgrade as upgrade_v0003

MIGRATIONS = (
    (1, "initial identity, sessions, registration, and audit tables", upgrade_v0001),
    (2, "workspaces, memberships, department grants, and default general workspace", upgrade_v0002),
    (3, "private conversations, messages, and source references", upgrade_v0003),
)
