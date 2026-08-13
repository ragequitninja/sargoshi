"""Speaker-embedding extraction.

Turns an utterance (16 kHz mono S16LE PCM - the same audio contract as STT) into
an L2-normalised voiceprint vector. Like the STT backend, model inference is
blocking and GIL-heavy, so it runs on a worker thread via `asyncio.to_thread`
and never stalls the event loop.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from speechbrain.inference.speaker import EncoderClassifier

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_INT16_FULL_SCALE = 32_768.0
# ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) emits 192-d embeddings.
_ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_ECAPA_DIM = 192


class EmbeddingError(ValueError):
    """Raised when an embedding cannot be produced (e.g. empty audio)."""


@runtime_checkable
class Embedder(Protocol):
    """Contract for a speaker-embedding model."""

    @property
    def model_id(self) -> str:
        """Stable id of the embedding model — voiceprints are tagged with it."""

    @property
    def dimension(self) -> int: ...

    @property
    def sample_rate(self) -> int: ...

    async def load(self) -> None: ...

    async def warm(self) -> None: ...

    async def embed(self, pcm: bytes) -> np.ndarray:
        """Return an L2-normalized (D,) float32 embedding for the PCM."""


class ECAPAEmbedder(Embedder):
    """SpeechBrain ECAPA-TDNN embedder.

    Runs on CPU by default (embeddings are cheap, ~10-40 ms, and this keeps VRAM
    for STT); pass ``device="cuda"`` to co-locate with the GPU STT path.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        savedir: str | None = None,
        source: str = _ECAPA_SOURCE,
    ) -> None:
        self._device = device
        self._runtime_device = "cpu"
        self._savedir = savedir
        self._source = source
        self._model: EncoderClassifier | None = None

    @property
    def model_id(self) -> str:
        return "ecapa-tdnn"

    @property
    def dimension(self) -> int:
        return _ECAPA_DIM

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def load(self) -> None:
        logger.info("Loading ECAPA embedder %r (device=%s)", self._source, self._device)
        self._model = await asyncio.to_thread(self._load_sync)
        logger.info("ECAPA embedder loaded")

    def _load_sync(self) -> EncoderClassifier:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        # Resolve the configured device to a concrete torch device now that torch
        # is importable (e.g. "auto" -> cuda/cpu, "rocm" -> cuda, GPU fallback).
        self._runtime_device = _resolve_runtime_device(self._device)
        logger.info("ECAPA embedder device: %s -> %s", self._device, self._runtime_device)
        return EncoderClassifier.from_hparams(
            source=self._source,
            savedir=self._savedir,
            run_opts={"device": self._runtime_device},
        )

    async def warm(self) -> None:
        self._require_model()
        logger.info("Warming ECAPA embedder")
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 s
        await asyncio.to_thread(self._embed_sync, silence)

    async def embed(self, pcm: bytes) -> np.ndarray:
        self._require_model()
        audio = pcm_to_float32(pcm)
        if audio.size == 0:
            raise EmbeddingError("cannot embed empty audio")
        return await asyncio.to_thread(self._embed_sync, audio)

    def _embed_sync(self, audio: np.ndarray) -> np.ndarray:
        import torch

        model = self._require_model()
        signal = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        signal = signal.to(self._runtime_device)
        with torch.no_grad():
            emb = model.encode_batch(signal)  # (1, 1, D)
        vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        return l2_normalize(vec)

    def _require_model(self) -> EncoderClassifier:
        if self._model is None:
            raise RuntimeError("ECAPAEmbedder used before load(); call load() first.")
        return self._model


def create_embedder(model: str, *, device: str = "cpu", savedir: str | None = None) -> Embedder:
    """Construct the speaker embedder named by `model` (config `speaker.model`).

    `device` is the configured `speaker.device` (cpu/cuda/rocm/openvino/auto). To
    add a new speaker-ID model: implement the `Embedder` protocol (including a
    stable, unique `model_id`) and route its config name to it below. Voiceprints
    are tagged with `model_id`, so models never cross-match.
    """
    m = model.strip().lower()
    if m in ("ecapa-tdnn", "ecapa", _ECAPA_SOURCE):
        dev = str(device).strip().lower()
        if dev not in SUPPORTED_DEVICES:
            raise ValueError(
                f"The ecapa-tdnn speaker embedder does not support device "
                f"{device!r}; supported devices: {', '.join(SUPPORTED_DEVICES)}."
            )
        return ECAPAEmbedder(device=dev, savedir=savedir)
    raise ValueError(f"Unknown speaker model {model!r}; expected 'ecapa-tdnn'.")


# Devices the ECAPA/SpeechBrain speaker embedder can actually run on. Config
# validates `speaker.device` against this — the same contract as the STT
# `backends.SUPPORTED_DEVICES` — so an unsupported device errors at config load
# instead of falling back or silently disabling the feature.
#
# ECAPA is PyTorch-based: cpu, cuda, rocm (torch's ROCm build exposes AMD GPUs as
# the "cuda" device), and auto. OpenVINO would need its own embedder backend, so
# it isn't offered here yet. When a model that supports more devices is added,
# device support becomes model-aware.
SUPPORTED_DEVICES = ("cpu", "cuda", "rocm", "auto")


def _resolve_runtime_device(device: str) -> str:
    """Resolve a configured device to the concrete torch device at load time.

    ``auto`` picks cuda when a GPU is visible, else cpu; ``rocm`` maps to the
    ``cuda`` string (PyTorch's ROCm build exposes AMD GPUs that way). An explicit
    ``cuda``/``rocm`` with no GPU present is left as-is so torch raises a clear
    error rather than silently running on CPU.
    """
    import torch

    d = str(device).strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "rocm":
        return "cuda"
    return d  # "cpu" or "cuda"


# ---------------------------------------------------------------------------
# Audio + vector helpers (pure numpy, no torch)
# ---------------------------------------------------------------------------


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    """16-bit signed little-endian PCM -> float32 in [-1, 1)."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _INT16_FULL_SCALE


def decode_wav(data: bytes) -> bytes:
    """Decode a WAV (any rate/channels/width) to 16 kHz mono S16LE PCM.

    Down-mixes to mono and linearly resamples to 16 kHz — the ASR-native format.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            rate = wav.getframerate()
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            raw = wav.readframes(wav.getnframes())
    except wave.Error as e:
        raise EmbeddingError(f"not a valid WAV file: {e}") from e

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise EmbeddingError(f"unsupported WAV sample width {width} bytes")

    arr = np.frombuffer(raw, dtype=dtype)
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)

    if width == 1:  # unsigned 8-bit
        samples = (arr.astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = arr.astype(np.float32) / _INT16_FULL_SCALE
    else:
        samples = arr.astype(np.float32) / 2147483648.0

    if rate != SAMPLE_RATE and samples.size:
        n_out = int(round(samples.size * SAMPLE_RATE / rate))
        x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.float32)

    pcm = np.clip(samples * _INT16_FULL_SCALE, -32768, 32767).astype("<i2")
    return pcm.tobytes()


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap 16-bit mono S16LE PCM in a WAV container (for stored audio)."""
    buf = io.BytesIO()
    with wave.Wave_write(buf) as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def pcm_duration_s(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    """Duration of S16LE mono PCM in seconds."""
    return (len(pcm) // 2) / sample_rate


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return `vec` scaled to unit L2 norm (float32)."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


def trim_silence(
    pcm: bytes,
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = 20,
    threshold: float = 0.01,
    pad_ms: int = 100,
) -> bytes:
    """Trim leading and trailing near-silence using a simple RMS gate.

    A conservative placeholder for a real VAD (silero/webrtcvad is the intended
    upgrade). It only trims the ends and pads back a margin, so it won't cut
    speech mid-utterance; if the whole clip is below threshold the original is
    returned unchanged.
    """
    samples = pcm_to_float32(pcm)
    frame = int(sample_rate * frame_ms / 1000)
    if frame <= 0 or samples.size < frame:
        return pcm

    n_frames = samples.size // frame
    framed = samples[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt((framed**2).mean(axis=1))
    voiced = np.where(rms >= threshold)[0]
    if voiced.size == 0:
        return pcm

    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, voiced[0] * frame - pad)
    end = min(samples.size, (voiced[-1] + 1) * frame + pad)
    out = np.clip(samples[start:end] * _INT16_FULL_SCALE, -32768, 32767)
    return out.astype("<i2").tobytes()
