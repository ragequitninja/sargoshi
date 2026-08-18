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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import WavLMForXVector

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_INT16_FULL_SCALE = 32_768.0
# ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) emits 192-d embeddings.
_ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_ECAPA_DIM = 192
# WavLM Base+ (SV) via transformers WavLMForXVector emits 512-d embeddings.
_WAVLM_SOURCE = "microsoft/wavlm-base-plus-sv"
_WAVLM_MODEL_ID = "wavlm-base-plus-sv"
_WAVLM_DIM = 512
# Torch execution devices shared by the PyTorch embedders (ECAPA, TorchWavLM).
_TORCH_DEVICES = ("cpu", "cuda", "rocm", "auto")
# speaker.device tokens for the OpenVINO WavLM backend -> OpenVINO runtime device.
_OPENVINO_TARGETS = {
    "openvino": "AUTO",
    "openvino:cpu": "CPU",
    "openvino:gpu": "GPU",
    "openvino:npu": "NPU",
}


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


class TorchWavLMEmbedder(Embedder):
    """WavLM Base+ (SV) speaker embedder on PyTorch.

    Transformers' ``WavLMForXVector`` head emits a 512-d x-vector and is markedly
    more robust on short utterances than ECAPA. Runs on CPU by default (keeps VRAM
    for STT); pass ``device="cuda"`` to co-locate with the GPU STT path.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        savedir: str | None = None,
        source: str = _WAVLM_SOURCE,
    ) -> None:
        self._device = device
        self._runtime_device = "cpu"
        self._savedir = savedir
        self._source = source
        self._model: WavLMForXVector | None = None
        self._feature_extractor: Any = None

    @property
    def model_id(self) -> str:
        return _WAVLM_MODEL_ID

    @property
    def dimension(self) -> int:
        return _WAVLM_DIM

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def load(self) -> None:
        logger.info("Loading WavLM embedder %r (device=%s)", self._source, self._device)
        self._model, self._feature_extractor = await asyncio.to_thread(self._load_sync)
        logger.info("WavLM embedder loaded")

    def _load_sync(self) -> tuple[WavLMForXVector, object]:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self._runtime_device = _resolve_runtime_device(self._device)
        logger.info("WavLM embedder device: %s -> %s", self._device, self._runtime_device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self._source, cache_dir=self._savedir)
        model = WavLMForXVector.from_pretrained(self._source, cache_dir=self._savedir)
        model.to(self._runtime_device).eval()
        return model, feature_extractor

    async def warm(self) -> None:
        self._require_model()
        logger.info("Warming WavLM embedder")
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
        inputs = self._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: tensor.to(self._runtime_device) for name, tensor in inputs.items()}
        with torch.no_grad():
            emb = model(**inputs).embeddings  # (1, D)
        vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        return l2_normalize(vec)

    def _require_model(self) -> WavLMForXVector:
        if self._model is None:
            raise RuntimeError("TorchWavLMEmbedder used before load(); call load() first.")
        return self._model


class OpenVINOWavLMEmbedder(Embedder):
    """WavLM Base+ (SV) speaker embedder on OpenVINO (Intel CPU/GPU/NPU).

    Runs the same fp32 weights as ``TorchWavLMEmbedder`` through the OpenVINO
    runtime via optimum-intel's ``OVModelForAudioXVector``. It reports the same
    ``model_id`` and produces embeddings in the same space as the torch backend, so
    voiceprints enrolled on one runtime match on the other. ``ov_device`` is the
    OpenVINO device string: ``AUTO`` (picks NPU/GPU/CPU), ``CPU``, ``GPU``, ``NPU``.
    """

    def __init__(
        self,
        *,
        ov_device: str = "AUTO",
        savedir: str | None = None,
        source: str = _WAVLM_SOURCE,
    ) -> None:
        self._ov_device = ov_device
        self._savedir = savedir
        self._source = source
        self._model: Any = None
        self._feature_extractor: Any = None

    @property
    def model_id(self) -> str:
        return _WAVLM_MODEL_ID

    @property
    def dimension(self) -> int:
        return _WAVLM_DIM

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def load(self) -> None:
        logger.info("Loading WavLM embedder %r on OpenVINO (device=%s)", self._source, self._ov_device)
        self._model, self._feature_extractor = await asyncio.to_thread(self._load_sync)
        logger.info("WavLM OpenVINO embedder loaded")

    def _load_sync(self):
        from optimum.intel import OVModelForAudioXVector
        from transformers import AutoFeatureExtractor

        feature_extractor = AutoFeatureExtractor.from_pretrained(self._source, cache_dir=self._savedir)
        # export=True converts the HF weights to OpenVINO IR on first load (cached
        # under savedir/HF_HOME); later loads reuse the IR.
        model = OVModelForAudioXVector.from_pretrained(
            self._source,
            export=True,
            device=self._ov_device,
            cache_dir=self._savedir,
        )
        return model, feature_extractor

    async def warm(self) -> None:
        self._require_model()
        logger.info("Warming WavLM OpenVINO embedder")
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 s
        await asyncio.to_thread(self._embed_sync, silence)

    async def embed(self, pcm: bytes) -> np.ndarray:
        self._require_model()
        audio = pcm_to_float32(pcm)
        if audio.size == 0:
            raise EmbeddingError("cannot embed empty audio")
        return await asyncio.to_thread(self._embed_sync, audio)

    def _embed_sync(self, audio: np.ndarray) -> np.ndarray:
        model = self._require_model()
        inputs = self._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        emb = model(**inputs).embeddings  # (1, D)
        arr = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
        return l2_normalize(arr.squeeze().astype(np.float32))

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("OpenVINOWavLMEmbedder used before load(); call load() first.")
        return self._model


def create_embedder(model: str, *, device: str = "cpu", savedir: str | None = None) -> Embedder:
    """Construct the speaker embedder for `model` on `device` (config `speaker.*`).

    The model family is resolved from `model` (`speaker.model`) and handed to its
    builder, which validates `device` (`speaker.device`) and returns the concrete
    embedder. Voiceprints are tagged with the embedder's `model_id`, so different
    models never cross-match.
    """
    family = _resolve_family(model)
    return _EMBEDDER_FACTORIES[family](device=device, savedir=savedir)


def _resolve_family(model: str) -> str:
    m = model.strip().lower()
    for family, aliases in _MODEL_ALIASES.items():
        if m in aliases:
            return family
    raise ValueError(f"Unknown speaker model {model!r}; expected one of {', '.join(_EMBEDDER_FACTORIES)}.")


def _build_ecapa(*, device: str, savedir: str | None) -> Embedder:
    return ECAPAEmbedder(device=_check_device(device, "ecapa-tdnn", _TORCH_DEVICES), savedir=savedir)


def _build_wavlm(*, device: str, savedir: str | None) -> Embedder:
    dev = str(device).strip().lower()
    if dev in _TORCH_DEVICES:
        _require_transformers()
        return TorchWavLMEmbedder(device=dev, savedir=savedir)
    if dev in _OPENVINO_TARGETS:
        _require_openvino()
        return OpenVINOWavLMEmbedder(ov_device=_OPENVINO_TARGETS[dev], savedir=savedir)
    supported = (*_TORCH_DEVICES, *_OPENVINO_TARGETS)
    raise ValueError(
        f"The {_WAVLM_MODEL_ID} speaker embedder does not support device "
        f"{device!r}; supported devices: {', '.join(supported)}."
    )


def _check_device(device: str, model_id: str, supported: tuple[str, ...]) -> str:
    dev = str(device).strip().lower()
    if dev not in supported:
        raise ValueError(
            f"The {model_id} speaker embedder does not support device "
            f"{device!r}; supported devices: {', '.join(supported)}."
        )
    return dev


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The wavlm speaker embedder needs the 'transformers' package; "
            "install the optional extra: pip install 'sargoshi[wavlm]'."
        ) from exc


def _require_openvino() -> None:
    try:
        from optimum.intel import OVModelForAudioXVector  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The wavlm OpenVINO backend needs 'optimum-intel[openvino]'; "
            "install the optional extra: pip install 'sargoshi[wavlm-openvino]'."
        ) from exc


# Config aliases accepted for each canonical model_id (what voiceprints are tagged with).
_MODEL_ALIASES = {
    "ecapa-tdnn": ("ecapa-tdnn", "ecapa", _ECAPA_SOURCE),
    _WAVLM_MODEL_ID: ("wavlm", _WAVLM_MODEL_ID, _WAVLM_SOURCE),
}

_EMBEDDER_FACTORIES = {
    "ecapa-tdnn": _build_ecapa,
    _WAVLM_MODEL_ID: _build_wavlm,
}


# Every `speaker.device` value config may accept — the union across all speaker
# models. ECAPA and the torch WavLM backend take the torch devices; the WavLM
# OpenVINO backend adds the `openvino*` targets. Config validates against this union
# so the value parses; the exact model x device constraint is enforced per family in
# `create_embedder` at load, which errors clearly if a model can't run on the chosen
# device (e.g. ECAPA cannot use `openvino`).
SUPPORTED_DEVICES = (*_TORCH_DEVICES, *_OPENVINO_TARGETS)


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
