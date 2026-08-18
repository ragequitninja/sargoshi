from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .embed import (
    ECAPAEmbedder,
    Embedder,
    EmbeddingError,
    OpenVINOWavLMEmbedder,
    TorchWavLMEmbedder,
    create_embedder,
    pcm_duration_s,
    trim_silence,
)
from .enrol import Enroller, EnrolmentError, EnrolResult
from .identify import Candidate, Identification, SpeakerIdentifier
from .store import EmbeddingInfo, StoredProfile, VoiceprintStore

if TYPE_CHECKING:
    from ..config import SpeakerConfig

logger = logging.getLogger(__name__)

__all__ = [
    "SpeakerService",
    "Embedder",
    "ECAPAEmbedder",
    "TorchWavLMEmbedder",
    "OpenVINOWavLMEmbedder",
    "create_embedder",
    "EmbeddingError",
    "VoiceprintStore",
    "StoredProfile",
    "EmbeddingInfo",
    "SpeakerIdentifier",
    "Identification",
    "Candidate",
    "Enroller",
    "EnrolResult",
    "EnrolmentError",
    "trim_silence",
    "pcm_duration_s",
]


class SpeakerService:
    def __init__(
        self,
        *,
        model: str,
        threshold: float,
        db_path: str,
        unknown_label: str = "<Unknown>",
        enabled: bool = True,
        device: str = "cpu",
        model_cache: str | None = None,
        top_k: int = 5,
    ) -> None:
        self._enabled = enabled
        self._unknown_label = unknown_label
        self._embedder = create_embedder(model, device=device, savedir=model_cache)
        # The store tags voiceprints with the embedder's model id and scopes
        # matching to it, so switching models never silently cross-matches.
        self._store = VoiceprintStore(db_path, embedding_model=self._embedder.model_id)
        self._identifier = SpeakerIdentifier(
            embedder=self._embedder,
            store=self._store,
            threshold=threshold,
            top_k=top_k,
        )
        self._enroller = Enroller(
            embedder=self._embedder,
            store=self._store,
            identifier=self._identifier,
        )
        self._started = False

    @classmethod
    def from_config(
        cls,
        cfg: SpeakerConfig,
        *,
        model_cache: str | None = None,
    ) -> SpeakerService:
        return cls(
            model=cfg.model,
            threshold=cfg.threshold,
            db_path=cfg.db_path,
            unknown_label=cfg.unknown_label,
            enabled=cfg.enabled,
            device=cfg.device,  # cpu / cuda / rocm / auto (speaker.device)
            model_cache=model_cache,
        )

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if not self._enabled:
            logger.info("Speaker ID disabled; SpeakerService.start() is a no-op")
            return
        await self._store.open()
        await self._embedder.load()
        await self._embedder.warm()
        # Recompute centroids for the active model (handles a model change) and
        # warn about profiles that have no voiceprints for it.
        active = await self._store.rebuild_centroids()
        stale = len(await self._store.list_profiles()) - active
        if stale > 0:
            logger.warning(
                "%d speaker profile(s) have no voiceprints for model %r — re-enroll them under this model to use them.",
                stale,
                self._embedder.model_id,
            )
        await self._identifier.refresh()
        self._started = True
        logger.info("SpeakerService ready (model=%s)", self._embedder.model_id)

    async def stop(self) -> None:
        await self._store.close()
        self._started = False

    # -- operations --------------------------------------------------------

    async def identify(self, pcm: bytes) -> Identification:
        if not self._enabled or not self._started:
            return Identification.unknown()
        return await self._identifier.identify(pcm)

    async def enrol(
        self,
        *,
        name: str,
        audio: list[bytes],
        attributes: dict[str, str] | None = None,
        speaker_id: str | None = None,
    ) -> EnrolResult:
        self._require_started()
        return await self._enroller.enrol(
            name=name,
            audio=audio,
            attributes=attributes,
            speaker_id=speaker_id,
        )

    async def list_speakers(self) -> list[StoredProfile]:
        self._require_started()
        return await self._store.list_profiles()

    async def get_speaker(self, speaker_id: str) -> StoredProfile | None:
        self._require_started()
        return await self._store.get_profile(speaker_id)

    async def list_embeddings(self, speaker_id: str) -> list[EmbeddingInfo]:
        self._require_started()
        return await self._store.list_embeddings(speaker_id)

    async def get_embedding_audio(self, embedding_id: int) -> bytes | None:
        self._require_started()
        return await self._store.get_embedding_audio(embedding_id)

    async def delete_embedding(self, embedding_id: int) -> str | None:
        """Delete one voiceprint; returns the affected speaker_id (or None)."""
        self._require_started()
        speaker_id = await self._store.delete_embedding(embedding_id)
        if speaker_id is not None:
            await self._identifier.refresh()
        return speaker_id

    async def update_speaker(
        self,
        speaker_id: str,
        *,
        name: str | None = None,
        attributes: dict[str, str] | None = None,
    ) -> bool:
        """Edit a profile's name/attributes (metadata only), then refresh the index."""
        self._require_started()
        updated = await self._store.update_profile(speaker_id, name=name, attributes=attributes)
        if updated:
            await self._identifier.refresh()
        return updated

    async def delete_speaker(self, speaker_id: str) -> bool:
        self._require_started()
        deleted = await self._store.delete_profile(speaker_id)
        if deleted:
            await self._identifier.refresh()
        return deleted

    async def similarity_matrix(self):
        self._require_started()
        return await self._identifier.similarity_matrix()

    # -- status ------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def model(self) -> str:
        """Active speaker-embedding model id — voiceprints are tagged with it."""
        return self._embedder.model_id

    @property
    def ready(self) -> bool:
        return self._started

    @property
    def unknown_label(self) -> str:
        return self._unknown_label

    def _require_started(self) -> None:
        if not self._enabled:
            raise RuntimeError("Speaker ID is disabled")
        if not self._started:
            raise RuntimeError("SpeakerService not started; call start() first.")
