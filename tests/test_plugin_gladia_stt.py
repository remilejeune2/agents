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
async def open_v2_stream(monkeypatch):
    """Opens STTv2 streams that only process the messages a test gives them."""
    from livekit.plugins.gladia import STTv2, stt_v2

    async def idle(self) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(stt_v2.SpeechStreamv2, "_run", idle)
    monkeypatch.setattr(stt_v2, "FINAL_WAIT_S", 0.05)
    streams = []

    def open_stream(**options):
        stream = STTv2(api_key="test-key", languages=["fr"], **options).stream()
        recorded = _Recorded()
        stream._event_ch = recorded
        streams.append(stream)
        return stream, recorded

    yield open_stream
    for stream in streams:
        await stream.aclose()


@pytest.fixture
async def v2_stream(open_v2_stream):
    return open_v2_stream()


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


def test_v2_session_sends_every_end_of_turn_option_under_gladia_s_names():
    from livekit.plugins.gladia import STTv2
    from livekit.plugins.gladia.stt_v2 import _session_config

    stt = STTv2(
        api_key="test-key",
        eot_threshold=0.85,
        eager_eot_threshold=0.6,
        min_end_of_turn_silence_ms=100,
        silence_ms=150,
        stop_s=1.0,
        reeval_ms=100,
        window_s=6.0,
        vad_threshold=0.6,
        speculation=True,
        speculation_threshold=0.9,
    )
    config = _session_config(stt._opts)

    assert config["end_of_turn"] == {
        "eot_threshold": 0.85,
        "eager_eot_threshold": 0.6,
        "min_end_of_turn_silence_when_confident": 100,
        "silence_ms": 150,
        "stop_s": 1.0,
        "reeval_ms": 100,
        "window_s": 6.0,
        "vad_threshold": 0.6,
        "speculation": {"threshold": 0.9},
    }
    assert config["messages_config"]["receive_realtime_processing_events"] is True
    assert config["messages_config"]["receive_speech_events"] is False


def test_v2_session_asks_for_speculation_with_gladia_s_threshold_by_default():
    from livekit.plugins.gladia import STTv2
    from livekit.plugins.gladia.stt_v2 import _session_config

    config = _session_config(STTv2(api_key="test-key", speculation=True)._opts)

    assert config["end_of_turn"] == {"speculation": True}
    assert config["messages_config"]["receive_realtime_processing_events"] is True
    default = _session_config(STTv2(api_key="test-key")._opts)
    assert default["messages_config"]["receive_realtime_processing_events"] is False


def test_v2_rejects_a_speculation_threshold_without_speculation():
    from livekit.plugins.gladia import STTv2

    with pytest.raises(ValueError, match="speculation_threshold"):
        STTv2(api_key="test-key", speculation_threshold=0.9)


def _speculative_turn(final: str, recommendation: str | None) -> list[dict]:
    messages = [
        _partial("je voudrais réserver une table"),
        {"type": "eager_end_of_turn", "data": {"text": "Je voudrais réserver une table"}},
        {"type": "end_of_turn", "data": {"reason": "model", "text": "Je voudrais réserver"}},
        _final(final),
    ]
    if recommendation:
        messages.append(
            {"type": "speculation", "data": {"results": [{"recommendation": recommendation}]}}
        )
    return messages


async def test_v2_ends_the_turn_on_the_preflight_text_when_gladia_keeps_the_early_reply(
    open_v2_stream,
):
    stream, recorded = open_v2_stream(eager_eot_threshold=0.6, speculation=True)
    for message in _speculative_turn("Je voudrais réserver une table, svp.", "keep"):
        stream._process_message(message)

    assert recorded()[-3:] == [
        ("PREFLIGHT_TRANSCRIPT", "Je voudrais réserver une table"),
        ("FINAL_TRANSCRIPT", "Je voudrais réserver une table"),
        ("END_OF_SPEECH", ""),
    ]


async def test_v2_ends_the_turn_on_the_final_when_gladia_discards_the_early_reply(
    open_v2_stream,
):
    stream, recorded = open_v2_stream(eager_eot_threshold=0.6, speculation=True)
    for message in _speculative_turn("Je voudrais réserver une table au Procope.", "discard"):
        stream._process_message(message)

    assert recorded()[-2:] == [
        ("FINAL_TRANSCRIPT", "Je voudrais réserver une table au Procope."),
        ("END_OF_SPEECH", ""),
    ]


async def test_v2_ends_the_turn_on_the_final_when_the_verdict_never_comes(open_v2_stream):
    stream, recorded = open_v2_stream(eager_eot_threshold=0.6, speculation=True)
    for message in _speculative_turn("Je voudrais réserver une table.", None):
        stream._process_message(message)
    assert recorded()[-1] == ("PREFLIGHT_TRANSCRIPT", "Je voudrais réserver une table")

    await asyncio.sleep(0.1)
    stream._process_message(_final("Merci."))

    # the next final opens a new turn: it isn't taken for this turn's late final
    assert recorded()[-4:] == [
        ("FINAL_TRANSCRIPT", "Je voudrais réserver une table."),
        ("END_OF_SPEECH", ""),
        ("START_OF_SPEECH", ""),
        ("INTERIM_TRANSCRIPT", "Merci."),
    ]


async def test_v2_without_speculation_ignores_the_text_of_an_eager_end_of_turn(v2_stream):
    stream, recorded = v2_stream
    for message in _speculative_turn("Je voudrais réserver une table, svp.", "keep"):
        stream._process_message(message)

    assert recorded()[-3:] == [
        ("PREFLIGHT_TRANSCRIPT", "je voudrais réserver une table"),
        ("FINAL_TRANSCRIPT", "Je voudrais réserver une table, svp."),
        ("END_OF_SPEECH", ""),
    ]


async def test_v2_a_repeated_end_of_turn_does_not_drop_the_next_turn_s_final(v2_stream):
    stream, recorded = v2_stream
    stream._process_message(_partial("Bonjour"))
    stream._process_message({"type": "end_of_turn", "data": {"reason": "model"}})
    stream._process_message({"type": "end_of_turn", "data": {"reason": "timeout"}})
    stream._process_message(_final("Bonjour."))
    await asyncio.sleep(0.1)
    stream._process_message(_final("Au revoir."))

    assert recorded() == [
        ("START_OF_SPEECH", ""),
        ("INTERIM_TRANSCRIPT", "Bonjour"),
        ("FINAL_TRANSCRIPT", "Bonjour."),
        ("END_OF_SPEECH", ""),
        ("START_OF_SPEECH", ""),
        ("INTERIM_TRANSCRIPT", "Au revoir."),
    ]


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


def test_v2_session_sends_audio_speaker_and_formatting_options():
    from livekit.plugins.gladia import STTv2
    from livekit.plugins.gladia.stt_v2 import _session_config

    default = _session_config(STTv2(api_key="test-key")._opts)
    for key in ("pre_processing", "diarization", "numerals", "emails", "addresses", "pii"):
        assert key not in default
    assert default["sample_rate"] == 16000

    stt = STTv2(
        api_key="test-key",
        sample_rate=8000,
        speech_threshold="auto",
        diarization=True,
        max_speakers=2,
        numerals="CH",
        emails=True,
        addresses={"country": "FR"},
        pii={"entities": "pci", "mode": "mask"},
    )
    config = _session_config(stt._opts)
    assert config["sample_rate"] == 8000
    assert config["pre_processing"] == {"speech_threshold": "auto"}
    assert config["diarization"] is True
    assert config["diarization_config"] == {"max_speakers": 2}
    assert config["numerals"] == {"region": "CH"}
    assert config["emails"] is True
    assert config["addresses"] == {"country": "FR"}
    assert config["pii"] == {"entities": "pci", "mode": "mask"}
    assert stt.capabilities.diarization is True
    assert _session_config(STTv2(api_key="test-key", numerals=True)._opts)["numerals"] is True


def test_v2_rejects_speaker_counts_without_diarization_and_unknown_rates():
    from livekit.plugins.gladia import STTv2

    with pytest.raises(ValueError, match="diarization"):
        STTv2(api_key="test-key", max_speakers=2)
    with pytest.raises(ValueError, match="sample_rate"):
        STTv2(api_key="test-key", sample_rate=22050)


async def test_v2_labels_a_turn_with_the_speaker_of_most_of_its_finals(v2_stream):
    stream, recorded = v2_stream

    def final(text: str, speaker: int) -> dict:
        message = _final(text)
        message["data"]["utterance"]["speaker"] = speaker
        return message

    for message in (
        final("Bonjour.", 1),
        final("Hm.", 0),
        final("Je voudrais un rendez-vous.", 1),
        {"type": "end_of_turn", "data": {}},
        final("Oui ?", 0),
        {"type": "end_of_turn", "data": {}},
    ):
        stream._process_message(message)

    finals = [e for e in recorded.events if e.type.name == "FINAL_TRANSCRIPT"]
    assert [(e.alternatives[0].text, e.alternatives[0].speaker_id) for e in finals] == [
        ("Bonjour. Hm. Je voudrais un rendez-vous.", "S1"),
        ("Oui ?", "S0"),
    ]


async def test_v2_keeps_for_a_reconnect_only_the_audio_after_the_last_final(v2_stream):
    stream, _ = v2_stream
    second = 16000 * 2
    stream._keep_for_replay(b"\x01\x00" * 16000 * 3)  # 3 s sent
    final = _final("Bonjour.")
    final["data"]["utterance"]["end"] = 2.0
    stream._process_message(final)

    assert len(stream._replay) == second
    assert stream._replay_start == 2 * second

    # a long silence after it: a new session hears 15 s at most
    stream._keep_for_replay(b"\x00\x00" * 16000 * 30)
    assert len(stream._replay) == 15 * second


class _ScriptedWebsocket:
    """A Gladia websocket that drops after its messages, or closes cleanly."""

    def __init__(self, messages, *, drop: bool):
        self.messages, self.drop, self.sent = messages, drop, []
        self.close_code = 1006 if drop else 1000

    async def send_bytes(self, data):
        self.sent.append(data)

    async def send_str(self, data):
        pass

    async def close(self):
        pass

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        import aiohttp

        for message in self.messages:
            yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(message))
        if not self.drop:
            await asyncio.Event().wait()


async def test_v2_a_dropped_connection_resumes_the_turn_in_a_new_session(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.gladia import STTv2

    first = _ScriptedWebsocket([_partial("Je voudrais")], drop=True)
    second = _ScriptedWebsocket(
        [_final("Je voudrais un rendez-vous."), {"type": "end_of_turn", "data": {}}], drop=False
    )
    sockets = iter([first, second])
    stream = STTv2(api_key="test-key").stream(
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
    )

    async def connect():
        return next(sockets)

    monkeypatch.setattr(stream, "_connect_ws", connect)
    stream._keep_for_replay(b"\x01\x00" * 1600)  # audio of the open utterance

    events = []

    async def collect():
        async for event in stream:
            events.append(
                (event.type.name, event.alternatives[0].text if event.alternatives else "")
            )
            if event.type.name == "END_OF_SPEECH":
                return

    await asyncio.wait_for(collect(), timeout=3)
    await stream.aclose()

    # the turn was not ended on the drop: the new session finished it
    assert events == [
        ("START_OF_SPEECH", ""),
        ("INTERIM_TRANSCRIPT", "Je voudrais"),
        ("INTERIM_TRANSCRIPT", "Je voudrais un rendez-vous."),
        ("FINAL_TRANSCRIPT", "Je voudrais un rendez-vous."),
        ("END_OF_SPEECH", ""),
    ]
    assert second.sent[0] == b"\x01\x00" * 1600
