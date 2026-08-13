"""Speaker identification: cosine match against enrolled centroids.

Given an utterance's PCM, extract an embedding and compare it (cosine) to
every enrolled profile's centroid; the argmax wins if it clears the threshold,
otherwise the speaker is "unknown". Centroids are held in an in-memory matrix
so the hot path is one matmul — no DB hit per identify - refreshed whenever
enrolment changes the store.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from .embed import Embedder
from .store import VoiceprintStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Candidate:
    speaker_id: str
    name: str
    similarity: float  # cosine similarity vs the profile centroid


@dataclass(slots=True)
class Identification:
    """Result of identifying an utterance."""

    known: bool
    speaker_id: str | None
    name: str | None
    confidence: float
    attributes: dict[str, str] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)

    @classmethod
    def unknown(cls, confidence: float = 0.0, candidates=None) -> Identification:
        return cls(
            known=False,
            speaker_id=None,
            name=None,
            confidence=confidence,
            attributes={},
            candidates=candidates or [],
        )


class SpeakerIdentifier:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VoiceprintStore,
        threshold: float,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._threshold = threshold
        self._top_k = top_k

        # In-memory index, rebuilt by refresh().
        self._ids: list[str] = []
        self._names: list[str] = []
        self._attrs: list[dict[str, str]] = []
        self._matrix: np.ndarray | None = None  # (N, D) normalized centroids
        self._lock = asyncio.Lock()

    async def refresh(self) -> None:
        """Rebuild the in-memory centroid index from the store."""
        profiles = await self._store.list_profiles()
        dim = self._embedder.dimension
        ids, names, attrs, centroids = [], [], [], []
        for p in profiles:
            if p.centroid is None:
                continue  # profile with no embeddings yet
            if p.centroid.shape[0] != dim:
                logger.warning(
                    "Skipping %s (%s): centroid dim %d != embedder dim %d (model changed?)",
                    p.id,
                    p.name,
                    p.centroid.shape[0],
                    dim,
                )
                continue
            ids.append(p.id)
            names.append(p.name)
            attrs.append(p.attributes)
            centroids.append(p.centroid)

        matrix = np.vstack(centroids).astype(np.float32) if centroids else None
        async with self._lock:
            self._ids, self._names, self._attrs, self._matrix = (
                ids,
                names,
                attrs,
                matrix,
            )
        logger.info("Speaker index refreshed: %d enrolled profile(s)", len(ids))

    async def identify(self, pcm: bytes) -> Identification:
        query = await self._embedder.embed(pcm)
        async with self._lock:
            matrix = self._matrix
            ids, names, attrs = self._ids, self._names, self._attrs

        if matrix is None or matrix.shape[0] == 0:
            return Identification.unknown()

        sims = matrix @ query  # (N,) cosine (both sides L2-normalized)
        candidates = self._rank(sims, ids, names)
        best_i = int(np.argmax(sims))
        best = float(sims[best_i])
        confidence = _confidence(best)

        if best >= self._threshold:
            return Identification(
                known=True,
                speaker_id=ids[best_i],
                name=names[best_i],
                confidence=confidence,
                attributes=dict(attrs[best_i]),
                candidates=candidates,
            )
        return Identification.unknown(confidence=confidence, candidates=candidates)

    def _rank(self, sims: np.ndarray, ids: list[str], names: list[str]) -> list[Candidate]:
        k = min(self._top_k, sims.shape[0])
        order = np.argsort(sims)[::-1][:k]
        return [Candidate(ids[i], names[i], float(sims[i])) for i in order.tolist()]

    async def similarity_matrix(self) -> tuple[list[str], np.ndarray]:
        """Pairwise cosine between enrolled centroids (UI collision warning)."""
        async with self._lock:
            matrix = self._matrix
            names = list(self._names)
        if matrix is None or matrix.shape[0] == 0:
            return [], np.zeros((0, 0), dtype=np.float32)
        return names, (matrix @ matrix.T)


def _confidence(similarity: float) -> float:
    """Clamp cosine similarity to a [0, 1] confidence (calibration is a TODO)."""
    return float(min(1.0, max(0.0, similarity)))
