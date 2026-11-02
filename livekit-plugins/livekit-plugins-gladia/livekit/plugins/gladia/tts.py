# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gladia's realtime text to speech, on the regional API (eu1, us1).

``POST /v2/tts/live`` opens a session and returns its WebSocket: the text goes in as it
is written, the audio comes back as binary PCM frames while it is synthesized
(https://eu1.docs.gladia.io/text-to-speech/quickstart/). Gladia cuts the text into
sentences itself, so tokens are sent as they come, without a tokenizer.

Each segment of a stream is said by a session of its own, ended by ``stop_synthesis``:
``end_session`` is the only message that says the last audio is out. The next session is
opened while the segment plays, so a reply does not wait for the handshake.
"""

from __future__ import annotations

import asyncio
import json
import os
import weakref
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSModels, V2Regions

NUM_CHANNELS = 1
SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
MIN_SPEED = 0.7
MAX_SPEED = 1.2

SESSION_MAX_IDLE_S = 45.0
"""How long an open session may wait for its text. Gladia ends a session that receives
nothing for 60 s (``idle_timeout``)."""


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    speed: float | None
    style: str | None
    language: str | None
    sample_rate: int
    max_chars: int | None
    base_url: str
    api_key: str


def _session_config(opts: _TTSOptions) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": opts.model,
        "voice": opts.voice,
        "sample_rate": opts.sample_rate,
        "encoding": "pcm_s16le",
        "output": "binary",
    }
    voice_settings: dict[str, Any] = {}
    if opts.speed is not None:
        voice_settings["speed"] = opts.speed
    if opts.style:
        voice_settings["style"] = opts.style
    if voice_settings:
        config["voice_settings"] = voice_settings
    if opts.language:
        config["language"] = opts.language
    if opts.max_chars is not None:
        config["segmentation"] = {"max_chars": opts.max_chars}
    return config


def _check_speed(speed: float | None) -> None:
    if speed is not None and not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}")


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "seldonia-1",
        voice: str = "default",
        speed: float | None = None,
        style: str | None = None,
        language: str | None = None,
        sample_rate: int = 24000,
        max_chars: int | None = None,
        region: V2Regions = "eu1",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Gladia TTS on the regional realtime API.

        Args:
            model: The model: ``"seldonia-1"`` (follows ``style``), ``"seldonia-1-small"``
                (ignores it) or ``"seldonia-multi-1"``. Defaults to ``"seldonia-1"``.
            voice: One of the model's preset voices (``"en-1"``, ``"en-2"``, ``"ja-1"``...).
                Defaults to ``"default"``, the model's first.
            speed: The speaking rate, from 0.7 to 1.2; 1.0 is the voice's own. Defaults to
                Gladia's (1.0).
            style: How to say it, in words (``"warm, reassuring"``), 200 characters at most.
                ``seldonia-1`` follows it.
            language: A language hint (``"en"``, ``"fr"``...). Defaults to automatic: a voice
                speaks other languages with its own accent.
            sample_rate: 8000, 16000, 22050, 24000, 44100 or 48000 Hz. Defaults to 24000.
            max_chars: The longest segment Gladia cuts the text into, from 20 to 1000
                characters. Defaults to Gladia's (250).
            region: The region of the account, ``"eu1"`` or ``"us1"``. Defaults to ``"eu1"``.
            api_key: The account's key for the regional API, with the ``tts`` scope.
                Defaults to ``GLADIA_API_KEY``.
            base_url: The session endpoint, overriding ``region``
                (``"https://eu1.staging.api.gladia.io/v2/tts/live"``).
            http_session: The HTTP session to use. Defaults to the agent's.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        gladia_api_key = api_key if is_given(api_key) else os.environ.get("GLADIA_API_KEY")
        if not gladia_api_key:
            raise ValueError("Gladia API key is required. Set GLADIA_API_KEY or pass api_key")
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")
        _check_speed(speed)

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            speed=speed,
            style=style,
            language=language,
            sample_rate=sample_rate,
            max_chars=max_chars,
            base_url=base_url
            if is_given(base_url)
            else f"https://{region}.api.gladia.io/v2/tts/live",
            api_key=gladia_api_key,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._closed = False
        # a session says one segment, then ends: the pool only keeps the next one open
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=SESSION_MAX_IDLE_S,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Gladia"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        async with session.post(
            self._opts.base_url,
            headers={"x-gladia-key": self._opts.api_key},
            json=_session_config(self._opts),
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as res:
            body = await res.text()
            if res.status != 201:
                try:
                    detail = json.loads(body)
                except json.JSONDecodeError:
                    detail = body
                message = detail.get("message") if isinstance(detail, dict) else None
                raise APIStatusError(
                    message=f"Gladia TTS session refused: {message or body}",
                    status_code=res.status,
                    body=detail,
                )
            created = json.loads(body)

        ws = await asyncio.wait_for(session.ws_connect(created["url"]), timeout)
        try:
            # an unknown voice or no free worker shows here, before any text is sent
            msg = await ws.receive(timeout=timeout)
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise APIStatusError(
                    "Gladia TTS session closed before it started",
                    status_code=ws.close_code or -1,
                    body=f"{msg.data=}",
                )
            data = json.loads(msg.data)
            if data.get("type") != "start_session":
                raise APIError(message="Gladia TTS session did not start", body=data)
        except BaseException:
            await ws.close()
            raise
        logger.debug("Gladia TTS session %s started", created.get("id"))
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        # closing the WebSocket ends the session (client_disconnected): nothing to send
        await ws.close()

    async def _speak(
        self,
        texts: AsyncIterable[str],
        output_emitter: tts.AudioEmitter,
        conn_options: APIConnectOptions,
        on_text: Callable[[], None] | None = None,
    ) -> None:
        """Say ``texts`` in a session of its own, until its ``end_session``."""
        ws = await self._pool.get(timeout=conn_options.timeout)
        input_done = asyncio.Event()

        async def send_task() -> None:
            async for text in texts:
                if on_text:
                    on_text()
                await ws.send_str(json.dumps({"type": "text", "text": text}))
            await ws.send_str(json.dumps({"type": "stop_synthesis"}))
            input_done.set()

        async def recv_task() -> None:
            while True:
                # while the LLM writes, Gladia has nothing to say: only wait on the audio
                # once the text is all in
                msg = await ws.receive(
                    timeout=conn_options.timeout if input_done.is_set() else None
                )
                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                    continue
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Gladia TTS connection closed before the end of the session",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                mtype = data.get("type")
                if mtype == "end_session":
                    if data.get("status") == "error":
                        raise APIError(message="Gladia TTS session ended on an error", body=data)
                    return
                if mtype == "error":
                    if data.get("code") == "invalid_message":
                        logger.warning("Gladia TTS ignored a message: %s", data)
                        continue
                    raise APIError(message="Gladia TTS error", body=data)
                if mtype == "segment_error":
                    logger.warning("Gladia TTS could not say a segment: %s", data)
                elif mtype == "enterprise.handoff":
                    logger.debug("Gladia TTS session moved to another worker: %s", data)

        tasks = [asyncio.create_task(send_task()), asyncio.create_task(recv_task())]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            self._pool.remove(ws)
            if not self._closed:
                self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        style: NotGivenOr[str | None] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Change the voice for the next segments; the one being said keeps its own."""
        if is_given(speed):
            _check_speed(speed)
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed
        if is_given(style):
            self._opts.style = style
        if is_given(language):
            self._opts.language = language
        # the voice is set when the session is created: the open one has the old voice
        self._pool.invalidate()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        self._closed = True
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        async def texts() -> AsyncIterable[str]:
            yield self._input_text

        await self._tts._speak(texts(), output_emitter, self._conn_options)
        output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        segments_ch = utils.aio.Chan[utils.aio.Chan[str]]()

        async def split_input() -> None:
            texts: utils.aio.Chan[str] | None = None
            async for data in self._input_ch:
                if isinstance(data, str):
                    if not data:
                        continue
                    if texts is None:
                        texts = utils.aio.Chan[str]()
                        segments_ch.send_nowait(texts)
                    texts.send_nowait(data)
                elif texts is not None:
                    texts.close()
                    texts = None
            if texts is not None:
                texts.close()
            segments_ch.close()

        async def run_segments() -> None:
            async for texts in segments_ch:
                output_emitter.start_segment(segment_id=utils.shortuuid())
                await self._tts._speak(
                    texts, output_emitter, self._conn_options, on_text=self._mark_started
                )
                output_emitter.end_segment()

        tasks = [asyncio.create_task(split_input()), asyncio.create_task(run_segments())]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
