import logging
from dataclasses import dataclass

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info

from sargoshi.config import Config
from sargoshi.frontends.wyoming import (
    WyomingAsrHandler,
    _build_info,
    _whisper_languages,
)
from sargoshi.speaker import Identification

# ---------------------------------------------------------------------------
# Fakes (wyoming-specific — the STT pool and speaker service call surface)
# ---------------------------------------------------------------------------


@dataclass
class FakeTranscription:
    text: str
    duration_seconds: float = 1.23
    real_time_factor: float = 0.1
    language: str = "en"


class FakeSTTPool:
    model_id = "large-v3-turbo"
    device = "cpu"
    compute_type = "int8"

    def __init__(self, text="hello world"):
        self._text = text
        self.calls = []

    async def transcribe(self, pcm, opts):
        self.calls.append((pcm, opts))
        return FakeTranscription(self._text)


class FakeIdentSpeaker:
    enabled = True

    def __init__(self, ident=None, error=None):
        self._ident = ident
        self._error = error
        self.calls = 0

    async def identify(self, _pcm):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._ident


class CapturingHandler(WyomingAsrHandler):
    """Records written events instead of sending them to a socket."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.written = []

    async def write_event(self, event):
        self.written.append(event)


def _handler(pool, speaker, config, info_event=None):
    if info_event is None:
        info_event = _build_info(pool).event()
    # AsyncEventHandler only stores reader/writer; we override write_event and
    # never run the socket loop, so None/None is safe.
    return CapturingHandler(info_event, pool, speaker, config, None, None)


def _with_delivery(config, modes):
    d = config.to_dict()
    d["speaker"]["delivery"] = list(modes)
    return Config.from_dict(d)


KNOWN = Identification(
    known=True,
    speaker_id="id01",
    name="Pieter",
    confidence=0.92,
    attributes={"gender": "Male"},
)
UNKNOWN = Identification.unknown(confidence=0.3)


async def _speak(handler, pcm=b"\x01\x00" * 800, rate=16_000):
    await handler.handle_event(AudioStart(rate=rate, width=2, channels=1).event())
    if pcm:
        await handler.handle_event(AudioChunk(rate=rate, width=2, channels=1, audio=pcm).event())
    await handler.handle_event(AudioStop().event())


# ---------------------------------------------------------------------------
# Control events
# ---------------------------------------------------------------------------


async def test_describe_returns_info(config):
    pool = FakeSTTPool()
    info_event = _build_info(pool).event()
    h = _handler(pool, None, config, info_event=info_event)
    assert await h.handle_event(Describe().event()) is True
    assert len(h.written) == 1 and h.written[0] is info_event
    assert Info.is_type(h.written[0].type)


async def test_transcribe_sets_language_without_replying(config):
    h = _handler(FakeSTTPool(), None, config)
    await h.handle_event(Transcribe(language="fr").event())
    assert h._language == "fr"
    assert h.written == []  # a Transcribe is not answered directly


async def test_transcribe_name_is_ignored(config):
    # Model selection is intentionally not client-driven.
    h = _handler(FakeSTTPool(), None, config)
    await h.handle_event(Transcribe(name="tiny", language=None).event())
    assert h._language == config.backend.language  # unchanged (null default)


# ---------------------------------------------------------------------------
# Audio flow
# ---------------------------------------------------------------------------


async def test_audio_flow_transcribes_on_stop(config):
    pool = FakeSTTPool(text="hello world")
    h = _handler(pool, None, config)
    await _speak(h, pcm=b"\x02\x00" * 1000)
    assert len(pool.calls) == 1  # transcribed exactly once, on AudioStop
    assert Transcript.from_event(h.written[-1]).text == "hello world"


async def test_empty_audio_emits_empty_transcript(config):
    pool = FakeSTTPool(text="unused")
    h = _handler(pool, None, config)
    await h.handle_event(AudioStart(rate=16_000, width=2, channels=1).event())
    await h.handle_event(AudioStop().event())
    assert pool.calls == []  # never transcribed
    assert Transcript.from_event(h.written[-1]).text == ""


async def test_buffer_resets_between_utterances(config):
    pool = FakeSTTPool(text="x")
    h = _handler(pool, None, config)
    await _speak(h, pcm=b"\x01\x00" * 10)
    await _speak(h, pcm=b"\x02\x00" * 20)
    # second utterance carries only its own 40 bytes, not the first's
    assert len(pool.calls) == 2 and len(pool.calls[1][0]) == 40


# ---------------------------------------------------------------------------
# Speaker delivery
# ---------------------------------------------------------------------------


async def test_metadata_delivery_known(config):
    cfg = _with_delivery(config, ["metadata"])
    h = _handler(FakeSTTPool("hi"), FakeIdentSpeaker(ident=KNOWN), cfg)
    await _speak(h)
    ev = h.written[-1]
    assert Transcript.from_event(ev).text == "hi"  # text NOT modified (no inline)
    assert ev.data["speaker"] == {
        "id": "id01",
        "name": "Pieter",
        "confidence": 0.92,
        "attributes": {"gender": "Male"},
    }


async def test_metadata_emitted_for_unknown_too(config):
    cfg = _with_delivery(config, ["metadata"])
    h = _handler(FakeSTTPool("hi"), FakeIdentSpeaker(ident=UNKNOWN), cfg)
    await _speak(h)
    meta = h.written[-1].data["speaker"]
    assert meta["id"] is None and meta["name"] is None and meta["confidence"] == 0.3


async def test_inline_delivery_known_prefixes_text(config):
    cfg = _with_delivery(config, ["inline"])
    h = _handler(FakeSTTPool("hello"), FakeIdentSpeaker(ident=KNOWN), cfg)
    await _speak(h)
    ev = h.written[-1]
    assert Transcript.from_event(ev).text == "<User - Pieter - Male> hello"
    assert "speaker" not in ev.data  # inline only, no metadata key


async def test_inline_delivery_unknown_uses_label(config):
    cfg = _with_delivery(config, ["inline"])
    h = _handler(FakeSTTPool("hello"), FakeIdentSpeaker(ident=UNKNOWN), cfg)
    await _speak(h)
    assert Transcript.from_event(h.written[-1]).text == "<Unknown> hello"


async def test_speaker_failure_does_not_break_transcript(config):
    cfg = _with_delivery(config, ["metadata"])
    h = _handler(FakeSTTPool("still works"), FakeIdentSpeaker(error=RuntimeError("boom")), cfg)
    await _speak(h)
    ev = h.written[-1]
    assert Transcript.from_event(ev).text == "still works"
    assert "speaker" not in ev.data  # identification failed -> no metadata


async def test_no_speaker_service_plain_transcript(config):
    cfg = _with_delivery(config, ["metadata"])
    h = _handler(FakeSTTPool("plain"), None, cfg)
    await _speak(h)
    ev = h.written[-1]
    assert Transcript.from_event(ev).text == "plain"
    assert "speaker" not in ev.data


async def test_disabled_speaker_is_not_consulted(config):
    cfg = _with_delivery(config, ["metadata"])
    spk = FakeIdentSpeaker(ident=KNOWN)
    spk.enabled = False
    h = _handler(FakeSTTPool("plain"), spk, cfg)
    await _speak(h)
    assert spk.calls == 0 and "speaker" not in h.written[-1].data


# ---------------------------------------------------------------------------
# Format handling + advertisement
# ---------------------------------------------------------------------------


async def test_non_16k_format_warns_once(config, caplog):
    h = _handler(FakeSTTPool(), None, config)
    with caplog.at_level(logging.WARNING):
        await h.handle_event(AudioStart(rate=8_000, width=2, channels=1).event())
        # a following chunk in the same bad format must not warn again
        await h.handle_event(AudioChunk(rate=8_000, width=2, channels=1, audio=b"\x00\x00").event())
    assert h._warned_format is True
    warnings = [r for r in caplog.records if "expected 16 kHz mono" in r.getMessage()]
    assert len(warnings) == 1


def test_build_info_advertises_program_and_model():
    info = _build_info(FakeSTTPool())
    assert info.asr[0].name == "sargoshi"
    assert info.asr[0].models[0].name == "large-v3-turbo"
    assert info.asr[0].models[0].installed is True


def test_whisper_languages_nonempty():
    langs = _whisper_languages()
    assert isinstance(langs, list) and "en" in langs
