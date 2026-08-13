"""Backend interface and engine-neutral result types.

Every STT engine implements the `Backend` protocol and returns a `Transcription`,
so the frontends and the speaker-ID module never have to know which engine ran.

Audio contract: backends receive canonical **16 kHz mono S16LE PCM**.
Decode or Resampling, if ever needed, happens upstream in the frontends.
Engines will only ever see PCM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class Word:
    """A single word with timing, populated when word timestamps are requested."""

    start: float
    end: float
    word: str
    prob: float


@dataclass(slots=True)
class Segment:
    """A decoded segment of the utterance."""

    start: float
    end: float
    text: str
    avg_logprob: float


@dataclass(slots=True)
class Transcription:
    """Engine-neutral transcription result returned to every frontend."""

    text: str
    language: str
    segments: list[Segment]
    words: list[Word] | None
    duration_seconds: float
    real_time_factor: float  # real-time factor = processing_time / duration_seconds (lower is faster)


@dataclass(slots=True)
class TranscribeOpts:
    """Per-request decode options, normalised across engines.
    Translation will be dependent on the engine's capabilities. (see `Capabilities`)

    * Whisper models support "translate" tasks from any audio language
      to english text and "transcribe" tasks from the native audio
      language to the same native text language.
    """

    language: str | None = None
    task: str = "transcribe"
    beam_size: int = 5
    word_timestamps: bool = False
    initial_prompt: str | None = None
    temperature: float = 0.0
    vad_filter: bool = False


@dataclass(slots=True)
class Capabilities:
    """What an engine can do so that frontends can advertise or reject features."""

    word_timestamps: bool
    translate: bool
    streaming: bool
    languages: list[str] | None = None


@runtime_checkable
class Backend(Protocol):
    """Contract every STT engine implements."""

    async def load(self, model_id: str, *, device: str, compute_type: str) -> None:
        """Load model weights. Blocking work is offloaded to a worker thread."""

    async def warm(self) -> None:
        """Run a dummy inference to JIT/allocate before the model goes active."""

    async def unload(self) -> None:
        """Release model resources (VRAM/RAM) so another model can be loaded."""

    async def transcribe(self, pcm: bytes, opts: TranscribeOpts) -> Transcription:
        """Transcribe 16 kHz mono S16LE PCM into a `Transcription`."""

    def capabilities(self) -> Capabilities:
        """Report engine capabilities (word ts? translate? streaming?)."""
