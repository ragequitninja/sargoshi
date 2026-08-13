from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

# A COMPLETE config (no-defaults schema: every key is required). Carries a few
# comments + an inline flow map so the comment-preserving write path can be
# tested against it.
COMPLETE_CONFIG = """\
# sargoshi test configuration
logging:
  level: INFO
backend:
  device: cpu            # my device choice
  model: large-v3-turbo  # switch me via the UI
  compute_type: int8
  language: null
  beam_size: 5
  word_timestamps: false
  cpu_threads: 0
  num_workers: 1
frontends:
  wyoming:
    enabled: true
    uri: tcp://0.0.0.0:10300
  ui: {enabled: true, host: 0.0.0.0, port: 8080}   # inline flow map
speaker:
  enabled: true
  model: ecapa-tdnn
  device: cpu
  threshold: 0.70        # match threshold
  db_path: speakers.db
  delivery: [metadata, structured]
  inline_template: "<User - {name} - {gender}>"
  unknown_label: "<Unknown>"
storage:
  model_cache: models
"""


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_text() -> str:
    return COMPLETE_CONFIG


@pytest.fixture
def config_path(tmp_path, config_text):
    """A temp config.yaml holding a complete config (fresh per test)."""
    p = tmp_path / "config.yaml"
    p.write_text(config_text)
    return p


@pytest.fixture
def config(config_path):
    from sargoshi.config import load_config

    return load_config(config_path)


@pytest.fixture
def config_service(config_path):
    from sargoshi.config import ConfigService, load_config

    return ConfigService(load_config(config_path), config_path)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

SR = 16_000


def _sine_pcm(freq, *, dur=1.0, amp=0.5, noise=0.0, seed=0):
    """A sine tone as 16 kHz 16-bit mono S16LE PCM (optionally with reproducible noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    if noise:
        x = x + noise * rng.standard_normal(x.size)
    x = np.clip(x, -1, 1)
    return (x * 32767).astype("<i2").tobytes()


def _wav(pcm, rate=SR, channels=1):
    import io
    import wave

    buf = io.BytesIO()
    with wave.Wave_write(buf) as w:  # Wave_write directly so type inference is unambiguous
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _sine_wav(freq, **kw):
    """A sine tone wrapped in a 16 kHz mono WAV container (what enroll ingests)."""
    return _wav(_sine_pcm(freq, **kw))


def _make_wav(seconds=0.5, freq=220):
    """A short 16 kHz WAV suitable for an upload field."""
    return _wav(_sine_pcm(freq, dur=seconds, amp=0.3))


def _stereo_wav(freq, rate=44_100, dur=1.0):
    """A 2-channel, 44.1 kHz WAV (to exercise downmix + resample on enroll)."""
    t = np.arange(int(rate * dur)) / rate
    mono = (0.4 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    stereo = np.repeat(mono, 2)  # interleaved L=R
    return _wav(stereo.tobytes(), rate=rate, channels=2)


class _Audio:
    """Namespace of audio generators exposed via the ``audio`` fixture."""

    SR = SR
    sine_pcm = staticmethod(_sine_pcm)
    sine_wav = staticmethod(_sine_wav)
    make_wav = staticmethod(_make_wav)
    stereo_wav = staticmethod(_stereo_wav)


@pytest.fixture
def audio():
    return _Audio


# ---------------------------------------------------------------------------
# Fakes — stand in for the warm STT pool and the speaker service in web/UI tests
# ---------------------------------------------------------------------------


class FakePool:
    device = "cpu"
    compute_type = "int8"

    def __init__(self, model_id="large-v3-turbo", ready=True):
        self.model_id = model_id
        self._ready = ready

    @property
    def is_ready(self):
        return self._ready

    def status(self):
        return {
            "model": self.model_id,
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded": True,
            "ready": self._ready,
        }

    async def switch_model(self, model):
        self.model_id = model  # pretend pre-warm + swap succeeded


@dataclass
class FakeProfile:
    id: str
    name: str
    embedding_count: int
    attributes: dict = field(default_factory=dict)
    models: tuple = ("ecapa-tdnn",)


@dataclass
class FakeResult:
    speaker_id: str
    name: str
    created: bool
    embedding_count: int


@dataclass
class FakeEmb:
    id: int
    speaker_id: str
    created_at: str
    audio_bytes: int
    model: str = "ecapa-tdnn"


class FakeSpeaker:
    """In-memory stand-in for SpeakerService covering the UI's call surface."""

    enabled = True
    ready = True
    model = "ecapa-tdnn"

    def __init__(self):
        self._profiles = {}
        self._embs = {}  # sid -> list of (emb_id, audio_bytes, raw)
        self._n = 0
        self._eid = 0

    async def list_speakers(self):
        return list(self._profiles.values())

    async def get_speaker(self, sid):
        return self._profiles.get(sid)

    async def enrol(self, *, name, audio, attributes=None, speaker_id=None):
        if speaker_id and speaker_id in self._profiles:
            sid, created = speaker_id, False
        else:
            self._n += 1
            sid, created = f"id{self._n:02d}", True
            self._profiles[sid] = FakeProfile(sid, name, 0, attributes or {})
            self._embs[sid] = []
        for raw in audio:
            self._eid += 1
            self._embs[sid].append((self._eid, len(raw), raw))
        self._profiles[sid].embedding_count = len(self._embs[sid])
        return FakeResult(sid, self._profiles[sid].name, created, self._profiles[sid].embedding_count)

    async def update_speaker(self, sid, *, name=None, attributes=None):
        p = self._profiles.get(sid)
        if p is None:
            return False
        if name is not None:
            p.name = name
        if attributes is not None:
            p.attributes = attributes
        return True

    async def delete_speaker(self, sid):
        self._embs.pop(sid, None)
        return self._profiles.pop(sid, None) is not None

    async def list_embeddings(self, sid):
        return [FakeEmb(eid, sid, "2026-01-01T00:00:00", size) for (eid, size, _raw) in self._embs.get(sid, [])]

    async def get_embedding_audio(self, eid):
        for lst in self._embs.values():
            for i, _size, raw in lst:
                if i == eid:
                    return raw
        return None

    async def delete_embedding(self, eid):
        for sid, lst in self._embs.items():
            for idx, (i, _size, _raw) in enumerate(lst):
                if i == eid:
                    lst.pop(idx)
                    self._profiles[sid].embedding_count = len(lst)
                    return sid
        return None


class FakeEmbedder:
    """Torch-free embedder: bins a tone's spectrum into an 8-d unit vector.

    Only the embedding *model* is faked — the store, identifier, enroller, and
    centroid math under test are the real code.
    """

    model_id = "fake-a"
    dimension = 8
    sample_rate = SR

    async def load(self):
        pass

    async def warm(self):
        pass

    async def embed(self, pcm):
        from sargoshi.speaker.embed import l2_normalize

        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if x.size == 0:
            raise ValueError("cannot embed empty audio")
        mag = np.abs(np.fft.rfft(x))
        feat = np.array([b.sum() for b in np.array_split(mag, 8)], dtype=np.float32)
        return l2_normalize(feat)


@pytest.fixture
def fake_pool():
    return FakePool()


@pytest.fixture
def fake_speaker():
    return FakeSpeaker()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
