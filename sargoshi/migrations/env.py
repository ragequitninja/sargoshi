"""Alembic environment — driven only via the programmatic API (no CLI).

`sargoshi.db.run_migrations` builds a `Config` in code and sets `script_location`
and `sqlalchemy.url` on it, so this env just reads that URL, builds a synchronous
engine, and runs the migrations online. `render_as_batch=True` keeps future
SQLite column ALTERs working — SQLite can't alter columns in place, so Alembic
emulates it via a copy-and-swap (harmless for other backends).
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool

from sargoshi.db import Base

# Config object populated by run_migrations() (script_location + sqlalchemy.url).
config = context.config

# Target schema for --autogenerate when developers create new revisions.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL against a URL, without a live DBAPI connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection (the startup path)."""
    url = config.get_main_option("sqlalchemy.url")
    if url is None:
        raise RuntimeError("Alembic config is missing 'sqlalchemy.url'")
    connectable = create_engine(url, poolclass=pool.NullPool)
    try:
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
