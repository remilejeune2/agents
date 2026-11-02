from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

pytestmark = pytest.mark.plugin("gladia")

SAMPLE_RATE = 16000
# 50 ms of PCM per binary frame
CHUNK = b"\x01\x00" * (SAMPLE_RATE // 20)


class FakeGladia:
    """Gladia's /v2/tts/live: says each text it is sent as one 50 ms chunk."""

    def __init__(self) -> None:
        self.configs: list[dict] = []
        self.texts: list[list[str]] = []
        self.refuse: tuple[int, dict] | None = None
        self.start_error: dict | None = None
        self.app = web.Application()
        self.app.router.add_post("/v2/tts/live", self.create)
        self.app.router.add_get("/ws/{n}", self.connect)
        self.server = TestServer(self.app)

    @property
    def url(self) -> str:
        return str(self.server.make_url("/v2/tts/live"))

    async def create(self, request: web.Request) -> web.Response:
        assert request.headers["x-gladia-key"] == "test-key"
        if self.refuse:
            status, body = self.refuse
            return web.json_response(body, status=status)
        self.configs.append(await request.json())
        url = self.server.make_url(f"/ws/{len(self.configs) - 1}").with_scheme("ws")
        return web.json_response({"id": "s", "created_at": "", "url": str(url)}, status=201)

    async def connect(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if self.start_error:
            await ws.send_json(self.start_error)
            await ws.send_json({"type": "end_session", "status": "error"})
            await ws.close()
            return ws
        await ws.send_json({"type": "start_session", "sample_rate": SAMPLE_RATE})
        texts: list[str] = []
        self.texts.append(texts)
        async for msg in ws:
            data = json.loads(msg.data)
            if data["type"] == "text":
                texts.append(data["text"])
                await ws.send_json({"type": "segment_start", "segment_id": len(texts) - 1})
                await ws.send_bytes(CHUNK)
                await ws.send_json({"type": "segment_end", "segment_id": len(texts) - 1})
            elif data["type"] == "stop_synthesis":
                await ws.send_json({"type": "end_session", "status": "done"})
                await ws.close()
        return ws


@pytest.fixture
async def gladia():
    fake = FakeGladia()
    await fake.server.start_server()
    yield fake
    await fake.server.close()


@pytest.fixture
async def make_tts(gladia):
    from livekit.plugins.gladia import TTS

    sessions: list[aiohttp.ClientSession] = []
    created = []

    def make(**options):
        session = aiohttp.ClientSession()
        sessions.append(session)
        tts = TTS(
            api_key="test-key",
            base_url=gladia.url,
            sample_rate=SAMPLE_RATE,
            http_session=session,
            **options,
        )
        created.append(tts)
        return tts

    yield make
    for tts in created:
        await tts.aclose()
    for session in sessions:
        await session.close()


async def test_stream_sends_tokens_as_they_come(gladia, make_tts):
    tts = make_tts(voice="en-2", speed=1.1, style="warm", language="en")
    stream = tts.stream()
    for token in ("Hello ", "there. ", "Bye."):
        stream.push_text(token)
    stream.flush()
    stream.end_input()

    frames = [ev async for ev in stream]
    await stream.aclose()

    assert gladia.texts[0] == ["Hello ", "there. ", "Bye."]
    assert gladia.configs[0] == {
        "model": "seldonia-1",
        "voice": "en-2",
        "sample_rate": SAMPLE_RATE,
        "encoding": "pcm_s16le",
        "output": "binary",
        "voice_settings": {"speed": 1.1, "style": "warm"},
        "language": "en",
    }
    audio_ms = sum(ev.frame.duration for ev in frames) * 1000
    assert round(audio_ms) == 150


async def test_next_session_is_open_before_the_next_reply(gladia, make_tts):
    tts = make_tts()
    async for _ in tts.synthesize("One."):
        pass
    # the session that said "One." ended: another one is opened for the next reply
    for _ in range(50):
        if len(gladia.texts) == 2:
            break
        await asyncio.sleep(0.01)
    assert len(gladia.configs) == 2

    async for _ in tts.synthesize("Two."):
        pass
    assert gladia.texts[1] == ["Two."]


async def test_refused_session_raises_status_error(gladia, make_tts):
    from livekit.agents import APIConnectOptions, APIStatusError

    gladia.refuse = (403, {"error": "tts_not_enabled", "message": "text to speech is off"})
    tts = make_tts()
    with pytest.raises(APIStatusError) as raised:
        async for _ in tts.synthesize(
            "Hi.", conn_options=APIConnectOptions(max_retry=0, timeout=2)
        ):
            pass
    assert raised.value.status_code == 403
    assert "text to speech is off" in raised.value.message


async def test_error_at_connect_fails_before_any_text(gladia, make_tts):
    from livekit.agents import APIConnectOptions, APIError

    gladia.start_error = {"type": "error", "code": "no_worker"}
    tts = make_tts()
    with pytest.raises(APIError):
        async for _ in tts.synthesize(
            "Hi.", conn_options=APIConnectOptions(max_retry=0, timeout=2)
        ):
            pass
    assert gladia.texts == []


def test_options_are_checked():
    from livekit.plugins.gladia import TTS

    with pytest.raises(ValueError, match="speed"):
        TTS(api_key="test-key", speed=1.5)
    with pytest.raises(ValueError, match="sample_rate"):
        TTS(api_key="test-key", sample_rate=32000)
    tts = TTS(api_key="test-key", region="us1")
    assert tts._opts.base_url == "https://us1.api.gladia.io/v2/tts/live"
