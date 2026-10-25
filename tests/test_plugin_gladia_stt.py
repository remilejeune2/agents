from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.plugin("gladia")


async def test_update_options_stores_region_for_new_streams():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key", region="eu-west")
    stt.update_options(region="us-west")

    assert stt._opts.region == "us-west"


async def test_update_options_rejects_translation_without_target_languages():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key")
    with pytest.raises(ValueError, match="translation_target_languages is required"):
        stt.update_options(translation_enabled=True, region="us-west")
    assert stt._opts.translation_config.enabled is False
    assert stt._opts.region == "eu-west"

    stt = STT(api_key="test-key", translation_enabled=True, translation_target_languages=["fr"])
    with pytest.raises(ValueError, match="translation_target_languages is required"):
        stt.update_options(translation_target_languages=[])
    assert stt._opts.translation_config.target_languages == ["fr"]

    stt.update_options(translation_enabled=False, translation_target_languages=[])
    assert stt._opts.translation_config.enabled is False


# STTv2: Gladia's regional realtime API, which ends the user's turn


class _Recorded:
    def __init__(self) -> None:
        self.events: list = []

    def send_nowait(self, event) -> None:
        self.events.append(event)

    def __call__(self) -> list[tuple[str, str]]:
        return [
            (e.type.name, e.alternatives[0].text if e.alternatives else "") for e in self.events
        ]


@pytest.fixture
async def v2_stream(monkeypatch):
    from livekit.plugins.gladia import STTv2, stt_v2

    async def idle(self) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(stt_v2.SpeechStreamv2, "_run", idle)
    monkeypatch.setattr(stt_v2, "FINAL_WAIT_S", 0.05)
    stream = STTv2(api_key="test-key", languages=["fr"]).stream()
    recorded = _Recorded()
    stream._event_ch = recorded
    yield stream, recorded
    await stream.aclose()


def _partial(text: str) -> dict:
    return {"type": "transcript", "data": {"is_final": False, "utterance": {"text": text}}}


def _final(text: str) -> dict:
    return {"type": "transcript", "data": {"is_final": True, "utterance": {"text": text}}}


async def test_v2_sends_one_final_per_turn_even_when_gladia_closes_a_piece_at_a_hesitation(
    v2_stream,
):
    stream, recorded = v2_stream
    # the order of Gladia's messages on a turn with a hesitation, trimmed
    for message in (
        {"type": "speech_start", "data": {}},
        _partial("Je voudrais"),
        _final("Je voudrais savoir."),
        {"type": "speech_end", "data": {}},
        _partial("combien de langues"),
        {"type": "eager_end_of_turn", "data": {"probability": 0.7}},
        {"type": "turn_resumed", "data": {}},
        _partial("combien de langues supporte Solaria"),
        {"type": "end_of_turn", "data": {"probability": 0.95, "reason": "model"}},
        _final("Combien de langues supporte Solaria ?"),
    ):
        stream._process_message(message)

    assert recorded() == [
        ("START_OF_SPEECH", ""),
        ("INTERIM_TRANSCRIPT", "Je voudrais"),
        ("INTERIM_TRANSCRIPT", "Je voudrais savoir."),
        ("INTERIM_TRANSCRIPT", "Je voudrais savoir. combien de langues"),
        ("PREFLIGHT_TRANSCRIPT", "Je voudrais savoir. combien de langues"),
        ("INTERIM_TRANSCRIPT", "Je voudrais savoir. combien de langues"),
        ("INTERIM_TRANSCRIPT", "Je voudrais savoir. combien de langues supporte Solaria"),
        ("FINAL_TRANSCRIPT", "Je voudrais savoir. Combien de langues supporte Solaria ?"),
        ("END_OF_SPEECH", ""),
    ]


async def test_v2_ends_a_turn_on_its_words_when_the_final_is_late_and_drops_that_final(
    v2_stream,
):
    stream, recorded = v2_stream
    stream._process_message(_partial("Le prix par heure"))
    stream._process_message({"type": "end_of_turn", "data": {"reason": "timeout"}})
    await asyncio.sleep(0.1)
    stream._process_message(_final("Le prix par heure ?"))
    stream._process_message(_partial("Merci"))

    assert recorded() == [
        ("START_OF_SPEECH", ""),
        ("INTERIM_TRANSCRIPT", "Le prix par heure"),
        ("FINAL_TRANSCRIPT", "Le prix par heure"),
        ("END_OF_SPEECH", ""),
        ("START_OF_SPEECH", ""),
        ("INTERIM_TRANSCRIPT", "Merci"),
    ]


async def test_v2_opens_no_turn_on_speech_gladia_transcribes_as_nothing(v2_stream):
    stream, recorded = v2_stream
    stream._process_message({"type": "speech_start", "data": {}})
    stream._process_message(_partial(""))
    stream._process_message({"type": "end_of_turn", "data": {"reason": "timeout"}})

    assert recorded() == []


def test_v2_session_asks_gladia_to_end_turns_and_sends_only_the_options_set():
    from livekit.plugins.gladia import STTv2
    from livekit.plugins.gladia.stt_v2 import _session_config

    default = STTv2(api_key="test-key")
    tuned = STTv2(
        api_key="test-key",
        model="trantoria-1-fast",
        languages=["fr", "en"],
        code_switching=True,
        region="us1",
        eot_threshold=0.85,
        eager_eot_threshold=0.6,
        custom_vocabulary=["Solaria"],
    )

    assert _session_config(default._opts)["finals"] == "end_of_turn"
    assert _session_config(default._opts)["end_of_turn"] is True
    assert "realtime_processing" not in _session_config(default._opts)
    config = _session_config(tuned._opts)
    assert config["end_of_turn"] == {"eot_threshold": 0.85, "eager_eot_threshold": 0.6}
    assert config["language_config"] == {"languages": ["fr", "en"], "code_switching": True}
    assert config["realtime_processing"]["custom_vocabulary_config"] == {"vocabulary": ["Solaria"]}
    assert tuned._opts.base_url == "https://us1.api.gladia.io/v2/live"


def test_v2_rejects_an_eager_threshold_above_the_end_of_turn_one():
    from livekit.plugins.gladia import STTv2

    with pytest.raises(ValueError, match="eager_eot_threshold"):
        STTv2(api_key="test-key", eager_eot_threshold=0.95)
    with pytest.raises(ValueError, match="eager_eot_threshold"):
        STTv2(api_key="test-key", eot_threshold=0.7, eager_eot_threshold=0.8)


class _GladiaWS:
    """Plays Gladia's messages once the audio is in, and records what the stream sends."""

    def __init__(self, messages: list[dict]) -> None:
        self.audio_bytes = 0
        self.texts: list[dict] = []
        self.close_code = None
        self._messages = messages
        self._stopped = asyncio.Event()

    async def send_bytes(self, data: bytes) -> None:
        self.audio_bytes += len(data)

    async def send_str(self, data: str) -> None:
        self.texts.append(json.loads(data))
        self._stopped.set()

    def __aiter__(self):
        return self._receive()

    async def _receive(self):
        import aiohttp

        await self._stopped.wait()
        for message in self._messages:
            yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(message))

    async def close(self) -> None:
        pass


async def test_v2_streams_the_audio_and_turns_gladia_messages_into_a_turn():
    from livekit import rtc
    from livekit.plugins.gladia import STTv2

    ws = _GladiaWS(
        [
            _partial("Bonjour"),
            {"type": "end_of_turn", "data": {"reason": "model"}},
            _final("Bonjour."),
        ]
    )
    stream = STTv2(api_key="test-key", languages=["fr"]).stream()

    async def connect():
        return ws

    stream._connect_ws = connect
    samples = 16000 // 2
    stream.push_frame(rtc.AudioFrame(b"\x00\x00" * samples, 16000, 1, samples))
    stream.end_input()
    events = [event async for event in stream]

    assert ws.audio_bytes == samples * 2
    assert ws.texts == [{"type": "stop_recording"}]
    usage = [e for e in events if e.type.name == "RECOGNITION_USAGE"]
    turn = [e for e in events if e.type.name != "RECOGNITION_USAGE"]
    assert sum(e.recognition_usage.audio_duration for e in usage) == pytest.approx(0.5)
    assert [(e.type.name, [a.text for a in e.alternatives]) for e in turn] == [
        ("START_OF_SPEECH", []),
        ("INTERIM_TRANSCRIPT", ["Bonjour"]),
        ("FINAL_TRANSCRIPT", ["Bonjour."]),
        ("END_OF_SPEECH", []),
    ]
