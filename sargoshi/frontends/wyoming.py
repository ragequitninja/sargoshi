"""Wyoming ASR frontend.

Drop-in replacement that speaks Home Assistant's `wyoming` protocol.
Each connection follows the Wyoming ASR flow:

    Describe       -> Info (advertise ASR program + model + languages)
    Transcribe     -> (optional) set language for this utterance
    AudioStart     -> begin buffering
    AudioChunk*    -> append PCM
    AudioStop      -> transcribe the buffer, emit Transcript

Speaker labels are delivered per SPEAKER_DELIVERY: `metadata` adds a `speaker`
key to the Transcript event (HA ignores unknown keys, so this is a trial test),
`inline` prefixes the text with the configured template. Model selection is
intentionally not exposed, so clients transcribe against whatever the operator
configured.

Inline template example:

    `<speaker: name> rest of transcribed text here`

Metadata example:

    ```
    {
      "type": "transcript",
      "data": {
        "text": "rest of transcribed text here",
        "speaker": {
          "id": "ec99dbe5106d",
          "name": "John",
          "confidence": 0.8982,
          "attributes": {
            "gender": "Male",
            "height": "175cm"
          }
        }
      }
    }
    ```
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from ..backends.base import TranscribeOpts

if TYPE_CHECKING:
    from ..config import Config
    from ..pool import ModelPool
    from ..speaker import Identification, SpeakerService

logger = logging.getLogger(__name__)

# Fallback if faster-whisper's language table can't be imported.
_FALLBACK_LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "nl",
    "pl",
    "ru",
    "uk",
    "cs",
    "sv",
    "no",
    "da",
    "fi",
    "tr",
    "ar",
    "he",
    "hi",
    "zh",
    "ja",
    "ko",
    "vi",
    "th",
]


async def run_wyoming(config: Config, pool: ModelPool, speaker: SpeakerService | None) -> None:
    """Serve the Wyoming ASR protocol until cancelled."""
    info_event = _build_info(pool).event()
    uri = config.frontends.wyoming.uri
    server = AsyncServer.from_uri(uri)
    logger.info("Wyoming ASR listening on %s (model=%s)", uri, pool.model_id)
    await server.run(partial(WyomingAsrHandler, info_event, pool, speaker, config))


def _build_info(pool: ModelPool) -> Info:
    attribution = Attribution(
        name="SYSTRAN / faster-whisper",
        url="https://github.com/SYSTRAN/faster-whisper",
    )
    model = AsrModel(
        name=pool.model_id,
        description=f"Whisper {pool.model_id} ({pool.device}/{pool.compute_type})",
        attribution=attribution,
        installed=True,
        languages=_whisper_languages(),
        version=None,
    )
    program = AsrProgram(
        name="sargoshi",
        description="Sargoshi speech-to-text",
        attribution=Attribution(name="sargoshi", url="https://github.com/sargoshi"),
        installed=True,
        models=[model],
        version="0.1.0",
    )
    return Info(asr=[program])


def _whisper_languages() -> list[str]:
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES

        return sorted(_LANGUAGE_CODES)
    except ImportError:
        return list(_FALLBACK_LANGUAGES)


class WyomingAsrHandler(AsyncEventHandler):
    """Per-connection handler. Buffers audio, transcribes on AudioStop."""

    def __init__(
        self,
        info_event: Event,
        pool: ModelPool,
        speaker: SpeakerService | None,
        config: Config,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._info_event = info_event
        self._pool = pool
        self._speaker = speaker
        self._config = config

        self._audio = bytearray()
        self._language = config.backend.language  # None = autodetect
        self._warned_format = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info_event)
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
            # transcribe.name (model) is intentionally ignored — see module docs.
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            self._audio = bytearray()
            self._check_format(start.rate, start.width, start.channels)
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._check_format(chunk.rate, chunk.width, chunk.channels)
            self._audio.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            await self._finish()
            return True

        return True

    async def _finish(self) -> None:
        pcm = bytes(self._audio)
        self._audio = bytearray()
        if not pcm:
            await self.write_event(Transcript(text="").event())
            return

        opts = TranscribeOpts(
            language=self._language,
            beam_size=self._config.backend.beam_size,
            word_timestamps=self._config.backend.word_timestamps,
        )
        result = await self._pool.transcribe(pcm, opts)
        text = result.text
        logger.info(
            "Transcript (%.2fs audio, rtf=%.2f, lang=%s): %s",
            result.duration_seconds,
            result.real_time_factor,
            result.language,
            text,
        )

        ident = None
        if self._speaker is not None and self._speaker.enabled:
            try:
                ident = await self._speaker.identify(pcm)
            except Exception as e:  # never let speaker-ID break STT delivery
                logger.warning("Speaker identification failed: %s", e)

        delivery = set(self._config.speaker.delivery)
        if ident is not None and "inline" in delivery:
            text = self._apply_inline(text, ident)

        event = Transcript(text=text).event()
        if ident is not None and "metadata" in delivery:
            # Emitted for unknown speakers too (known=false) so consumers can see
            # the decision + confidence, not just successful matches.
            event.data["speaker"] = _speaker_metadata(ident)

        await self.write_event(event)

    def _apply_inline(self, text: str, ident: Identification) -> str:
        if ident.known and ident.name:
            fields = {"name": ident.name, **ident.attributes}
            tag = self._config.speaker.inline_template.format_map(_SafeDict(fields))
        else:
            tag = self._config.speaker.unknown_label
        return f"{tag} {text}".strip()

    def _check_format(self, rate: int, width: int, channels: int) -> None:
        if (rate, width, channels) != (16000, 2, 1) and not self._warned_format:
            self._warned_format = True
            logger.warning(
                "Received audio at %d Hz/%d-byte/%dch; expected 16 kHz mono "
                "S16LE. Passing through without conversion — transcription may "
                "be wrong. Have the client send 16 kHz mono 16-bit.",
                rate,
                width,
                channels,
            )


def _speaker_metadata(ident: Identification) -> dict:
    return {
        "id": ident.speaker_id,
        "name": ident.name,
        "confidence": round(ident.confidence, 4),
        "attributes": ident.attributes,
    }


class _SafeDict(dict):
    """format_map helper: unknown template fields render empty rather than error."""

    def __missing__(self, key: str) -> str:
        return ""
