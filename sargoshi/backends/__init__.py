from __future__ import annotations

from typing import Any

from .base import (
    Backend,
    Capabilities,
    Segment,
    TranscribeOpts,
    Transcription,
    Word,
)

__all__ = [
    "Backend",
    "Capabilities",
    "Segment",
    "Transcription",
    "TranscribeOpts",
    "Word",
    "create_backend",
    "SUPPORTED_DEVICES",
]


# Devices the backend STT layer supports. The configuration validates
# the backend.device against this. Currently supported backends are:
# CTranslate2/faster-whisper (cpu/cuda/auto)
SUPPORTED_DEVICES = ("cpu", "cuda", "auto")


def create_backend(device: str, **kwargs: Any) -> Backend:
    """Constructs the STT backend for `device` by lazy loading the library."""

    d = device.lower()
    if d in ("cuda", "cpu", "auto"):
        from .ctranslate2 import CTranslate2Backend

        return CTranslate2Backend(**kwargs)
    raise ValueError(f"Unknown device {device!r}; expected one of " f"'cuda', 'cpu', 'auto', 'rocm', 'openvino'.")
