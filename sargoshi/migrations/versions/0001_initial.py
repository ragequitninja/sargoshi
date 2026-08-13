from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "speakers",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("attributes", sa.JSON(), server_default="{}", nullable=False),
        sa.Column("dim", sa.Integer(), nullable=True),
        sa.Column("centroid", sa.LargeBinary(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("speaker_id", sa.String(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("audio", sa.LargeBinary(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["speaker_id"], ["speakers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_embeddings_speaker_id", "embeddings", ["speaker_id"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_embeddings_speaker_id", table_name="embeddings")
    op.drop_table("embeddings")
    op.drop_table("speakers")
