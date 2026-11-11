"""Create private, server-persisted conversation history."""

from sqlalchemy.engine import Connection

from manufacturing_rag.auth.models import Conversation, ConversationMessage


def upgrade(connection: Connection) -> None:
    Conversation.__table__.create(bind=connection, checkfirst=True)
    ConversationMessage.__table__.create(bind=connection, checkfirst=True)
