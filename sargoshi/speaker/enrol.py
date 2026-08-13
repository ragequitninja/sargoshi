"""Enrolment: turn uploaded audio into a stored voiceprint profile.

Accepts one or more WAV files in any format. Per file: decode and convert to
16 kHz mono 16-bit (the ASR-native format) -> VAD-trim -> embed. The **converted**
16 kHz mono audio (not the original upload) is stored together with the vector,
so we can re-embed or play it back later, and the profile centroid is recomputed.
Enrolling under a name that already exists just adds the new voiceprints to that profile.

Coordinates the embedder, the store, and — so the in-memory match index stays
current — the identifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .embed import Embedder, decode_wav, pcm_duration_s, pcm_to_wav, trim_silence
from .identify import SpeakerIdentifier
from .store import VoiceprintStore

logger = logging.getLogger(__name__)


class EnrolmentError(ValueError):
    """Raised when enrolment cannot proceed (no usable audio, bad id, ...)."""


@dataclass(slots=True)
class EnrolResult:
    speaker_id: str
    name: str
    created: bool
    embedding_count: int


class Enroller:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VoiceprintStore,
        identifier: SpeakerIdentifier | None = None,
        min_duration_s: float = 0.4,
        trim: bool = True,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._identifier = identifier
        self._min_duration_s = min_duration_s
        self._trim = trim

    async def enrol(
        self,
        *,
        name: str,
        audio: list[bytes],
        attributes: dict[str, str] | None = None,
        speaker_id: str | None = None,
    ) -> EnrolResult:
        """Enrol audio files under `speaker_id`, else an existing `name`, else new.

        Each item of `audio` is the raw uploaded file (WAV). The original bytes
        are stored alongside each embedding.
        """
        if not audio:
            raise EnrolmentError("no audio provided")

        items: list[tuple[np.ndarray, bytes]] = []  # (vector, converted 16k mono WAV)
        for i, raw in enumerate(audio):
            try:
                pcm = decode_wav(raw)  # convert to 16 kHz mono S16LE
            except Exception as e:
                logger.warning("Skipping file %d: could not decode WAV (%s)", i, e)
                continue
            trimmed = trim_silence(pcm) if self._trim else pcm
            duration = pcm_duration_s(trimmed)
            if duration < self._min_duration_s:
                logger.warning(
                    "Skipping file %d: %.2fs < min %.2fs",
                    i,
                    duration,
                    self._min_duration_s,
                )
                continue
            vector = await self._embedder.embed(trimmed)
            # Store the CONVERTED 16 kHz mono 16-bit audio, not the original.
            items.append((vector, pcm_to_wav(pcm)))

        if not items:
            raise EnrolmentError("no usable audio after decoding / trimming / duration filtering")

        # Resolve the target profile: explicit id > existing name > new profile.
        created = False
        if speaker_id is not None:
            profile = await self._store.get_profile(speaker_id)
            if profile is None:
                raise EnrolmentError(f"unknown speaker_id {speaker_id!r}")
        else:
            profile = await self._store.get_by_name(name)
            if profile is None:
                profile = await self._store.create_profile(name=name, attributes=attributes or {})
                created = True
        if attributes and not created:
            await self._store.set_attributes(profile.id, {**profile.attributes, **attributes})

        count = await self._store.add_embeddings(profile.id, items)

        # Keep the identifier's in-memory centroid index in sync.
        if self._identifier is not None:
            await self._identifier.refresh()

        result_name = name if created else profile.name
        logger.info(
            "Enrolled %s (%s): +%d voiceprints, %d total%s",
            profile.id,
            result_name,
            len(items),
            count,
            "" if created else " (linked to existing profile)",
        )
        return EnrolResult(
            speaker_id=profile.id,
            name=result_name,
            created=created,
            embedding_count=count,
        )
