"""The shared warm model pool.

One process, one asyncio loop, one warm STT model shared *by reference* across
every enabled frontend. The pool owns the backend lifecycle (load, warm,
hot-swap, unload) and is the single object every frontend calls to transcribe.

Concurrency comes from the backend (e.g. CTranslate2's ``num_workers`` plus GIL
release during inference), not from N model copies to avoid duplicating the model
in RAM or VRAM per frontend.
"""

from __future__ import annotations

import asyncio
import logging

from .backends import create_backend
from .backends.base import Backend, Capabilities, TranscribeOpts, Transcription

logger = logging.getLogger(__name__)


class ModelPool:
    """Owns the single warm STT backend and its lifecycle.

    ``backend_kwargs`` are forwarded to the selected backend constructor
    (e.g. ``download_root``, ``cpu_threads``, ``num_workers`` for CTranslate2).
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        compute_type: str,
        **backend_kwargs: object,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._compute_type = compute_type
        self._backend_kwargs = backend_kwargs

        self._backend: Backend | None = None
        self._warm = False
        # Serializes model swaps so two switches (or a switch + stop) can't race.
        self._switch_lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Load and warm the configured model. Call once at boot."""
        async with self._switch_lock:
            self._backend = await self._load_and_warm(self._model_id, self._compute_type)
            self._warm = True
        logger.info(
            "Model pool ready: %s on %s (%s)",
            self._model_id,
            self._device,
            self._compute_type,
        )

    async def switch_model(self, model_id: str, *, compute_type: str | None = None) -> None:
        """Pre-warm a new model, then atomically swap it in (zero-downtime).

        The current model keeps serving while the new one loads and warms; only
        once the new one is warm, do we swap the reference and unload the old one.
        """
        compute_type = compute_type or self._compute_type
        async with self._switch_lock:
            logger.info("Switching model -> %s (%s)", model_id, compute_type)
            new = await self._load_and_warm(model_id, compute_type)
            old = self._backend
            # Attribute assignment is atomic: in-flight transcriptions finish on
            # `old`; new requests hit `new`.
            self._backend = new
            self._model_id = model_id
            self._compute_type = compute_type
            self._warm = True
            if old is not None:
                await old.unload()
        logger.info("Active model is now %s (%s)", model_id, compute_type)

    async def stop(self) -> None:
        """Unload the active model and release its resources."""
        async with self._switch_lock:
            if self._backend is not None:
                await self._backend.unload()
            self._backend = None
            self._warm = False

    async def _load_and_warm(self, model_id: str, compute_type: str) -> Backend:
        backend = create_backend(self._device, **self._backend_kwargs)
        await backend.load(model_id, device=self._device, compute_type=compute_type)
        await backend.warm()
        return backend

    # -- inference ---------------------------------------------------------

    async def transcribe(self, pcm: bytes, opts: TranscribeOpts | None = None) -> Transcription:
        """Transcribe 16 kHz mono S16LE PCM on the active backend."""
        backend = self._require_backend()
        return await backend.transcribe(pcm, opts or TranscribeOpts())

    def capabilities(self) -> Capabilities:
        return self._require_backend().capabilities()

    # -- status ------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def is_ready(self) -> bool:
        """True, once the model is loaded AND warm. This drives GET /ready."""
        return self._backend is not None and self._warm

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    def status(self) -> dict[str, object]:
        """Snapshot for the status tile and health endpoints."""
        return {
            "model": self._model_id,
            "device": self._device,
            "compute_type": self._compute_type,
            "loaded": self.is_loaded,
            "ready": self.is_ready,
        }

    def _require_backend(self) -> Backend:
        if self._backend is None:
            raise RuntimeError("ModelPool used before start(); call start() first.")
        return self._backend
