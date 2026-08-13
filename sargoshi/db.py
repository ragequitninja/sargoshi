from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

from sqlalchemy import JSON, ForeignKey, Integer, LargeBinary, String, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

__all__ = [
    "Base",
    "Speaker",
    "Embedding",
    "async_url",
    "sync_url",
    "ensure_parent_dir",
    "make_async_engine",
    "run_migrations",
]

_MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent / "migrations"


class Base(DeclarativeBase):
    pass


class Speaker(Base):
    """A speaker profile with a derived, L2-normalized centroid for matching."""

    __tablename__ = "speakers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    attributes: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, server_default="{}")
    dim: Mapped[int | None] = mapped_column(Integer)  # embedding dimension
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)  # float32, normalized
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)

    embeddings: Mapped[list[Embedding]] = relationship(
        back_populates="speaker",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Embedding(Base):
    """One voiceprint: the vector, the converted source audio, and its model tag."""

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    speaker_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("speakers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # float32, normalized
    audio: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # 16 kHz mono 16-bit WAV
    model: Mapped[str] = mapped_column(String, nullable=False)  # embedder model id (e.g. ecapa-tdnn)
    created_at: Mapped[str] = mapped_column(String, nullable=False)

    speaker: Mapped[Speaker] = relationship(back_populates="embeddings")


# ---------------------------------------------------------------------------
# Engine / URL helpers
# ---------------------------------------------------------------------------


def async_url(db_path: str) -> str:
    """Runtime async SQLAlchemy URL for a file-backed SQLite ``db_path``."""
    return f"sqlite+aiosqlite:///{db_path}"


def sync_url(db_path: str) -> str:
    """Synchronous URL used by the Alembic migration engine at startup."""
    return f"sqlite:///{db_path}"


def ensure_parent_dir(db_path: str) -> None:
    """Create the parent directory for the file-backed DB if it's missing."""
    parent = pathlib.Path(db_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def make_async_engine(db_path: str) -> AsyncEngine:
    """Build the async engine for a file-backed DB, enabling SQLite FK enforcement."""
    from sqlalchemy.ext.asyncio import create_async_engine

    ensure_parent_dir(db_path)
    engine = create_async_engine(async_url(db_path))
    _enable_sqlite_foreign_keys(engine)
    return engine


def _enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """Turn on ``PRAGMA foreign_keys`` per connection so ON DELETE CASCADE works."""

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragma(dbapi_connection, _record):  # noqa: ANN001 (SQLAlchemy signature)
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# ---------------------------------------------------------------------------
# Migrations (programmatic Alembic — no CLI)
# ---------------------------------------------------------------------------


def run_migrations(db_path: str) -> None:
    """Bring the database schema to head at startup via Alembic's Python API."""

    from alembic import command
    from alembic.config import Config

    ensure_parent_dir(db_path)
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", sync_url(db_path))
    command.upgrade(cfg, "head")
    logger.info("Voiceprint DB %s migrated", db_path)
