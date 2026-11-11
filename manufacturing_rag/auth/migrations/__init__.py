"""Versioned schema migrations for persistent authentication data."""

from manufacturing_rag.auth.migrations.v0001_identity import upgrade as upgrade_v0001

MIGRATIONS = ((1, "initial identity, sessions, registration, and audit tables", upgrade_v0001),)
