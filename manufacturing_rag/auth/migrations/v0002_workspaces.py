from sqlalchemy import insert, select
from sqlalchemy.engine import Connection

from manufacturing_rag.auth.models import Base, Workspace, utcnow


def upgrade(connection: Connection) -> None:
    Base.metadata.create_all(bind=connection)
    exists = connection.execute(
        select(Workspace.id).where(Workspace.general_marker == "general").limit(1)
    ).first()
    if exists:
        return
    now = utcnow()
    connection.execute(
        insert(Workspace).values(
            id="campus-default",
            name="校园知识库",
            description="所有已登录用户可检索的通用工作区",
            workspace_type="general",
            general_marker="general",
            status="active",
            created_by=None,
            owner_user_id=None,
            version=1,
            created_at=now,
            updated_at=now,
        )
    )
