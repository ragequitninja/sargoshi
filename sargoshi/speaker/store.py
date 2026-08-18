from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.sql.functions import count as sa_count

from ..db import Base, Embedding, Speaker, make_async_engine
from .embed import l2_normalize

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingInfo:
    """Metadata for one stored embedding (audio kept out of the listing)."""

    id: int
    speaker_id: str
    created_at: str
    audio_bytes: int  # size of the stored source audio, 0 if none
    model: str = ""  # embedder model id that produced the vector


@dataclass(slots=True)
class StoredProfile:
    """A speaker profile as held in the store (embeddings summarized as centroid)."""

    id: str
    name: str
    attributes: dict[str, str] = field(default_factory=dict)
    dim: int | None = None
    centroid: np.ndarray | None = None  # (D,) normalized, None until first enroll
    embedding_count: int = 0
    models: tuple[str, ...] = ()  # distinct embedder model ids across its voiceprints
    created_at: str = ""
    updated_at: str = ""


class VoiceprintStore:
    def __init__(self, db_path: str, *, embedding_model: str) -> None:
        self._path = db_path
        self._embedding_model = embedding_model  # tags new voiceprints; scopes matching
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------

    async def open(self, *, create_schema: bool = False) -> None:
        from sqlalchemy.ext.asyncio import async_sessionmaker

        async with self._lock:
            if self._engine is None:
                self._engine = make_async_engine(self._path)
                self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
                if create_schema:
                    async with self._engine.begin() as conn:
                        await conn.run_sync(Base.metadata.create_all)
        logger.info("Voiceprint store open: %s", self._path)

    async def close(self) -> None:
        async with self._lock:
            engine, self._engine, self._sessionmaker = self._engine, None, None
        if engine is not None:
            await engine.dispose()

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise RuntimeError("VoiceprintStore is not open; call open() first.")
        async with self._lock:
            async with self._sessionmaker() as session:
                yield session

    # -- writes ------------------------------------------------------------

    async def create_profile(
        self,
        *,
        name: str,
        attributes: dict[str, str] | None = None,
        speaker_id: str | None = None,
    ) -> StoredProfile:
        sid = speaker_id or uuid.uuid4().hex[:12]
        attrs = attributes or {}
        now = _now()
        async with self._session() as session:
            session.add(
                Speaker(
                    id=sid,
                    name=name,
                    attributes=attrs,
                    dim=None,
                    centroid=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.commit()
        return StoredProfile(id=sid, name=name, attributes=attrs, created_at=now, updated_at=now)

    async def add_embeddings(self, speaker_id: str, items: list[tuple[np.ndarray, bytes]]) -> int:
        """Append embeddings (vector and original audio), recompute the centroid.

        `items` is a list of ``(vector, audio_bytes)`` where `audio_bytes` is the
        converted 16 kHz mono WAV kept for re-embedding/playback. Returns the
        speaker's new total embedding count for the active model.
        """
        if not items:
            raise ValueError("add_embeddings() requires at least one item")
        now = _now()
        model = self._embedding_model
        async with self._session() as session:
            exists = (await session.execute(select(Speaker.id).where(Speaker.id == speaker_id))).scalar_one_or_none()
            if exists is None:
                raise KeyError(f"unknown speaker_id {speaker_id!r}")
            session.add_all(
                Embedding(
                    speaker_id=speaker_id,
                    vector=_vec_to_blob(vector),
                    audio=audio,
                    model=model,
                    created_at=now,
                )
                for vector, audio in items
            )
            await session.flush()
            count = await _recompute_centroid(session, speaker_id, now, model)
            await session.commit()
            return count

    async def delete_embedding(self, embedding_id: int) -> str | None:
        """Delete one embedding and recompute its speaker's centroid.

        Returns the affected speaker_id (so callers can refresh the match index),
        or None if the embedding did not exist.
        """
        now = _now()
        async with self._session() as session:
            speaker_id = (
                await session.execute(select(Embedding.speaker_id).where(Embedding.id == embedding_id))
            ).scalar_one_or_none()
            if speaker_id is None:
                return None
            await session.execute(delete(Embedding).where(Embedding.id == embedding_id))
            await session.flush()
            await _recompute_centroid(session, speaker_id, now, self._embedding_model)
            await session.commit()
            return speaker_id

    async def rebuild_centroids(self) -> int:
        """Recompute every speaker's centroid for the active model.

        Called on startup so that after a speaker-model change, centroids reflect
        the active model and speakers with no voiceprints for it go NULL (match
        nobody) until re-enrolled. Returns the number of profiles that still have
        a usable centroid.
        """
        now = _now()
        async with self._session() as session:
            ids = (await session.execute(select(Speaker.id))).scalars().all()
            active = 0
            for sid in ids:
                if await _recompute_centroid(session, sid, now, self._embedding_model) > 0:
                    active += 1
            await session.commit()
            return active

    async def set_attributes(self, speaker_id: str, attributes: dict[str, str]) -> None:
        now = _now()
        async with self._session() as session:
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(Speaker).where(Speaker.id == speaker_id).values(attributes=attributes, updated_at=now)
                ),
            )
            if result.rowcount == 0:
                raise KeyError(f"unknown speaker_id {speaker_id!r}")
            await session.commit()

    async def update_profile(
        self,
        speaker_id: str,
        *,
        name: str | None = None,
        attributes: dict[str, str] | None = None,
    ) -> bool:
        """Update a profile's name and/or attributes; True if the profile existed.

        Embeddings and centroid are untouched — this is metadata only.
        """
        if name is None and attributes is None:
            return await self.get_profile(speaker_id) is not None
        values: dict[str, object] = {"updated_at": _now()}
        if name is not None:
            values["name"] = name
        if attributes is not None:
            values["attributes"] = attributes
        async with self._session() as session:
            result = cast(
                "CursorResult[Any]",
                await session.execute(update(Speaker).where(Speaker.id == speaker_id).values(**values)),
            )
            await session.commit()
            return result.rowcount > 0

    async def delete_profile(self, speaker_id: str) -> bool:
        async with self._session() as session:
            result = cast(
                "CursorResult[Any]",
                await session.execute(delete(Speaker).where(Speaker.id == speaker_id)),
            )
            await session.commit()
            return result.rowcount > 0

    # -- reads -------------------------------------------------------------

    async def get_profile(self, speaker_id: str) -> StoredProfile | None:
        async with self._session() as session:
            speaker = await session.get(Speaker, speaker_id)
            if speaker is None:
                return None
            count = await _embedding_count(session, speaker_id)
            models = await _distinct_models(session, speaker_id)
            return _to_profile(speaker, count, models)

    async def list_profiles(self) -> list[StoredProfile]:
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(
                        Speaker,
                        sa_count(Embedding.id),
                        func.group_concat(Embedding.model.distinct()),
                    )
                    .outerjoin(Embedding, Embedding.speaker_id == Speaker.id)
                    .group_by(Speaker.id)
                    .order_by(Speaker.name)
                )
            ).all()
            return [_to_profile(speaker, count, _parse_models(models)) for speaker, count, models in rows]

    async def get_embeddings(self, speaker_id: str) -> list[np.ndarray]:
        async with self._session() as session:
            blobs = (
                (
                    await session.execute(
                        select(Embedding.vector).where(Embedding.speaker_id == speaker_id).order_by(Embedding.id)
                    )
                )
                .scalars()
                .all()
            )
            return [_blob_to_vec_required(b) for b in blobs]

    async def get_by_name(self, name: str) -> StoredProfile | None:
        """Find a profile by exact name (used to link re-enrollment)."""
        async with self._session() as session:
            speaker = (
                (
                    await session.execute(
                        select(Speaker).where(Speaker.name == name).order_by(Speaker.created_at).limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if speaker is None:
                return None
            count = await _embedding_count(session, speaker.id)
            models = await _distinct_models(session, speaker.id)
            return _to_profile(speaker, count, models)

    async def list_embeddings(self, speaker_id: str) -> list[EmbeddingInfo]:
        """List a speaker's embeddings (metadata only — audio not loaded)."""
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(
                        Embedding.id,
                        Embedding.created_at,
                        func.length(Embedding.audio),
                        Embedding.model,
                    )
                    .where(Embedding.speaker_id == speaker_id)
                    .order_by(Embedding.id)
                )
            ).all()
            return [
                EmbeddingInfo(
                    id=r[0],
                    speaker_id=speaker_id,
                    created_at=r[1],
                    audio_bytes=r[2] or 0,
                    model=r[3] or "",
                )
                for r in rows
            ]

    async def get_embedding_audio(self, embedding_id: int) -> bytes | None:
        """Return the original audio stored for an embedding, or None."""
        async with self._session() as session:
            return (
                await session.execute(select(Embedding.audio).where(Embedding.id == embedding_id))
            ).scalar_one_or_none()


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


async def _embedding_count(session: AsyncSession, speaker_id: str) -> int:
    return (
        await session.execute(select(sa_count()).select_from(Embedding).where(Embedding.speaker_id == speaker_id))
    ).scalar_one()


async def _distinct_models(session: AsyncSession, speaker_id: str) -> tuple[str, ...]:
    """Sorted set of embedder model ids present in a speaker's voiceprints."""
    rows = (
        (await session.execute(select(Embedding.model).where(Embedding.speaker_id == speaker_id).distinct()))
        .scalars()
        .all()
    )
    return tuple(sorted(rows))


def _parse_models(concat: str | None) -> tuple[str, ...]:
    """Turn SQLite's ``group_concat(DISTINCT model)`` output into a sorted tuple."""
    if not concat:
        return ()
    return tuple(sorted({m for m in concat.split(",") if m}))


async def _recompute_centroid(session: AsyncSession, speaker_id: str, now: str, model: str) -> int:
    """Recompute a speaker's centroid from its ACTIVE-model voiceprints.

    Only embeddings tagged with `model` count. Voiceprints from a different
    embedding model live in a different space and are ignored. Returns how many
    active-model voiceprints the speaker has.
    """
    blobs = (
        (
            await session.execute(
                select(Embedding.vector)
                .where(Embedding.speaker_id == speaker_id, Embedding.model == model)
                .order_by(Embedding.id)
            )
        )
        .scalars()
        .all()
    )
    if blobs:
        matrix = np.vstack([_blob_to_vec_required(b) for b in blobs])
        centroid = l2_normalize(matrix.mean(axis=0))
        await session.execute(
            update(Speaker)
            .where(Speaker.id == speaker_id)
            .values(
                centroid=_vec_to_blob(centroid),
                dim=int(centroid.shape[0]),
                updated_at=now,
            )
        )
    else:  # last embedding removed — profile stays but matches nothing
        await session.execute(
            update(Speaker).where(Speaker.id == speaker_id).values(centroid=None, dim=None, updated_at=now)
        )
    return len(blobs)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec_required(blob: bytes) -> np.ndarray:
    """Decode a NOT-NULL float32 BLOB (embedding vectors)."""
    return np.frombuffer(blob, dtype=np.float32)


def _blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    """Decode a nullable float32 BLOB (centroids, None until first enroll)."""
    return None if blob is None else _blob_to_vec_required(blob)


def _to_profile(speaker: Speaker, embedding_count: int, models: tuple[str, ...] = ()) -> StoredProfile:
    return StoredProfile(
        id=speaker.id,
        name=speaker.name,
        attributes=speaker.attributes or {},
        dim=speaker.dim,
        centroid=_blob_to_vec(speaker.centroid),
        embedding_count=embedding_count,
        models=models,
        created_at=speaker.created_at,
        updated_at=speaker.updated_at,
    )
