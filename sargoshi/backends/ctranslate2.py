"""CTranslate2 / Faster-Whisper backend.

The primary NVIDIA engine (with universal CPU fallback), used by the `:cuda`
and `:cpu` image tags. Faster-Whisper wraps CTranslate2, which releases the GIL
during inference so that blocking `transcribe` calls run on a worker thread via
`asyncio.to_thread` while the event loop keeps serving the other frontends.

Supported devices: `cuda`, `cpu`, `auto`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from .base import Backend, Capabilities, Segment, TranscribeOpts, Transcription, Word

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# One second of silence @ 16 kHz used to warm the model.
_WARM_SAMPLES = 16_000
# S16LE full-scale, for int16 -> float32 [-1, 1) normalisation.
_INT16_FULL_SCALE = 32_768.0


class CTranslate2Backend(Backend):
    """Faster-Whisper (CTranslate2) implementation of the `Backend` protocol.

    Reference shares a single loaded model across all frontends. Set
    `num_workers` > 1 to allow that many `transcribe` calls to run truly
    concurrently in CTranslate2's internal pool. `cpu_threads` tunes intra-op
    parallelism on the CPU path.
    """

    def __init__(
        self,
        *,
        download_root: str | None = None,
        cpu_threads: int = 0,
        num_workers: int = 1,
    ) -> None:
        self._download_root = download_root
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers

        self._model: WhisperModel | None = None
        self._model_id: str | None = None
        self._device: str | None = None
        self._compute_type: str | None = None

    # -- lifecycle

    async def load(self, model_id: str, *, device: str, compute_type: str) -> None:
        device = self._normalize_device(device)
        logger.info(
            "Loading CTranslate2 model %r (device=%s, compute_type=%s)",
            model_id,
            device,
            compute_type,
        )

        self._model = await asyncio.to_thread(self._load_sync, model_id, device, compute_type)
        self._model_id = model_id
        self._device = device
        self._compute_type = compute_type
        logger.info("Model %r loaded", model_id)

    def _load_sync(self, model_id: str, device: str, compute_type: str) -> WhisperModel:
        from faster_whisper import WhisperModel

        return WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
            download_root=self._download_root,
            cpu_threads=self._cpu_threads,
            num_workers=self._num_workers,
        )

    async def warm(self) -> None:
        model = self._require_model()
        logger.info("Warming model %r", self._model_id)
        silence = np.zeros(_WARM_SAMPLES, dtype=np.float32)
        await asyncio.to_thread(self._warm_sync, model, silence)
        logger.info("Model %r warm", self._model_id)

    @staticmethod
    def _warm_sync(model: WhisperModel, audio: np.ndarray) -> None:
        segments, _info = model.transcribe(audio, beam_size=1, language="en")
        for _ in segments:
            pass

    async def unload(self) -> None:
        """Drop the model so its VRAM/RAM can be reclaimed before a switch."""
        self._model = None
        self._model_id = None
        self._device = None
        self._compute_type = None

    # -- inference

    async def transcribe(self, pcm: bytes, opts: TranscribeOpts) -> Transcription:
        model = self._require_model()
        audio = self._pcm_to_float32(pcm)
        if audio.size == 0:
            return Transcription(
                text="",
                language=opts.language or "",
                segments=[],
                words=[] if opts.word_timestamps else None,
                duration_seconds=0.0,
                real_time_factor=0.0,
            )

        return await asyncio.to_thread(self._transcribe_sync, model, audio, opts)

    def _transcribe_sync(self, model: WhisperModel, audio: np.ndarray, opts: TranscribeOpts) -> Transcription:
        started = time.perf_counter()
        segment_iter, info = model.transcribe(
            audio,
            language=opts.language or None,
            task=opts.task,
            beam_size=opts.beam_size,
            temperature=opts.temperature,
            initial_prompt=opts.initial_prompt,
            word_timestamps=opts.word_timestamps,
            vad_filter=opts.vad_filter,
        )

        segments: list[Segment] = []
        words: list[Word] | None = [] if opts.word_timestamps else None
        text_parts: list[str] = []

        for seg in segment_iter:
            segments.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    avg_logprob=seg.avg_logprob,
                )
            )
            text_parts.append(seg.text)
            if words is not None and seg.words:
                words.extend(Word(start=w.start, end=w.end, word=w.word, prob=w.probability) for w in seg.words)

        elapsed = time.perf_counter() - started
        duration_s = float(info.duration)
        rtf = elapsed / duration_s if duration_s > 0 else 0.0

        return Transcription(
            text="".join(text_parts).strip(),
            language=info.language,
            segments=segments,
            words=words,
            duration_seconds=duration_s,
            real_time_factor=rtf,
        )

    # -- introspection

    def capabilities(self) -> Capabilities:
        return Capabilities(word_timestamps=True, translate=True, streaming=False)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def compute_type(self) -> str | None:
        return self._compute_type

    # -- helpers

    def _require_model(self) -> WhisperModel:
        if self._model is None:
            raise RuntimeError("CTranslate2Backend used before load(); call load() first.")
        return self._model

    @staticmethod
    def _pcm_to_float32(pcm: bytes) -> np.ndarray:
        """16-bit signed little-endian PCM -> float32 in [-1, 1)."""
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _INT16_FULL_SCALE

    @staticmethod
    def _normalize_device(device: str) -> str:
        d = device.lower()
        if d in ("cuda", "cpu", "auto"):
            return d
        raise ValueError(
            f"CTranslate2Backend supports device 'cuda', 'cpu', or 'auto', got "
            f"{device!r}. CTranslate2 has no ROCm/OpenVINO support — use the "
            f"whisper.cpp (:rocm) or OpenVINO (:openvino) backend for that hardware."
        )
