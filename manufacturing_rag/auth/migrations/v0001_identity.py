from sqlalchemy.engine import Connection

from manufacturing_rag.auth.models import Base


def upgrade(connection: Connection) -> None:
    Base.metadata.create_all(bind=connection)
