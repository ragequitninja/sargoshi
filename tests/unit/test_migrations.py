import sqlite3

import numpy as np

from sargoshi.db import run_migrations
from sargoshi.speaker.store import VoiceprintStore


def _tables(db_file):
    con = sqlite3.connect(db_file)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def _alembic_version(db_file):
    con = sqlite3.connect(db_file)
    try:
        return con.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    finally:
        con.close()


def test_fresh_db_gets_schema_and_version(tmp_path):
    db = str(tmp_path / "sub" / "voice.db")  # nested — parent auto-created
    run_migrations(db)
    assert {"speakers", "embeddings", "alembic_version"} <= _tables(db)
    assert _alembic_version(db) == "0001"


def test_re_running_is_idempotent(tmp_path):
    db = str(tmp_path / "voice.db")
    run_migrations(db)
    run_migrations(db)  # no-op on an already-managed DB
    assert _alembic_version(db) == "0001"


async def test_store_opens_on_migrated_db_without_create_schema(tmp_path):
    db = str(tmp_path / "voice.db")
    run_migrations(db)  # production path: migrate first, then open the store

    store = VoiceprintStore(db, embedding_model="ecapa-tdnn")
    await store.open()  # create_schema defaults False — relies on the migration
    prof = await store.create_profile(name="Pieter", attributes={"role": "resident"})
    n = await store.add_embeddings(prof.id, [(np.ones(192, dtype=np.float32), b"RIFFfake")])
    assert n == 1
    got = await store.get_by_name("Pieter")
    assert got is not None and got.embedding_count == 1 and got.centroid is not None
    await store.close()
