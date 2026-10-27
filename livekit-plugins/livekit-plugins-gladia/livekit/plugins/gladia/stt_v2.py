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

"""Gladia's regional realtime API (eu1, us1), with Gladia's own end-of-turn detection.

Gladia decides the end of a turn from how the user speaks, not from silence alone
(https://eu1.docs.gladia.io/api-reference/). Use it with ``turn_detection="stt"``; the
stream translates Gladia's turn messages the way the Deepgram Flux plugin translates
Flux's:

    first words of a turn        START_OF_SPEECH
    partial transcript           INTERIM_TRANSCRIPT, the turn so far
    final inside the turn        INTERIM_TRANSCRIPT, the turn so far
    eager_end_of_turn            PREFLIGHT_TRANSCRIPT: the reply can start early
    turn_resumed                 INTERIM_TRANSCRIPT: the early reply is dropped
    end_of_turn (+ its final)    FINAL_TRANSCRIPT of the whole turn, then END_OF_SPEECH
    speculation (keep)           the FINAL_TRANSCRIPT repeats the preflight's text, so the
                                 early reply is kept

There is one final per turn, never one per pause: with STT turn detection a final that
arrives while the VAD hears silence commits the turn, so a final Gladia closes at a
hesitation would end the turn exactly where Gladia says it goes on.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._utils import PeriodicCollector
from .log import logger
from .models import V2Models, V2Regions

SAMPLE_RATE = 16000
AUDIO_CHUNK_MS = 100
DEFAULT_EOT_THRESHOLD = 0.9

FINAL_WAIT_S = 0.5
"""How long the end of a turn waits for the final of its last words. Gladia sends
``end_of_turn`` before that final (about 0.25 s measured): without the wait the turn
would end on the partial, unpunctuated and short of its last correction."""


@dataclass
class STTOptions:
    model: V2Models | str
    languages: list[str]
    code_switching: bool
    base_url: str
    eot_threshold: NotGivenOr[float] = NOT_GIVEN
    eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN
    min_end_of_turn_silence_ms: NotGivenOr[int] = NOT_GIVEN
    silence_ms: NotGivenOr[int] = NOT_GIVEN
    stop_s: NotGivenOr[float] = NOT_GIVEN
    reeval_ms: NotGivenOr[int] = NOT_GIVEN
    window_s: NotGivenOr[float] = NOT_GIVEN
    vad_threshold: NotGivenOr[float] = NOT_GIVEN
    speculation: bool = False
    speculation_threshold: NotGivenOr[float] = NOT_GIVEN
    custom_vocabulary: list[str] = field(default_factory=list)


# STTOptions field -> Gladia's end_of_turn field
_END_OF_TURN_FIELDS = {
    "eot_threshold": "eot_threshold",
    "eager_eot_threshold": "eager_eot_threshold",
    "min_end_of_turn_silence_ms": "min_end_of_turn_silence_when_confident",
    "silence_ms": "silence_ms",
    "stop_s": "stop_s",
    "reeval_ms": "reeval_ms",
    "window_s": "window_s",
    "vad_threshold": "vad_threshold",
}


def _session_config(opts: STTOptions) -> dict[str, Any]:
    """The body of ``POST /v2/live``. Gladia refuses any field it doesn't know, so
    only options that were set are sent."""
    end_of_turn: dict[str, Any] = {
        gladia_field: getattr(opts, name)
        for name, gladia_field in _END_OF_TURN_FIELDS.items()
        if is_given(getattr(opts, name))
    }
    if opts.speculation:
        end_of_turn["speculation"] = (
            {"threshold": opts.speculation_threshold}
            if is_given(opts.speculation_threshold)
            else True
        )

    config: dict[str, Any] = {
        "encoding": "wav/pcm",
        "sample_rate": SAMPLE_RATE,
        "bit_depth": 16,
        "channels": 1,
        "model": opts.model,
        "language_config": {"languages": opts.languages, "code_switching": opts.code_switching},
        "finals": "end_of_turn",
        "end_of_turn": end_of_turn or True,
        "messages_config": {
            "receive_partial_transcripts": True,
            "receive_final_transcripts": True,
            # a turn opens on its first words: speech_start and speech_end aren't used
            "receive_speech_events": False,
            "receive_acknowledgments": False,
            "receive_post_processing_events": False,
            # the speculation verdict is a realtime processing event
            "receive_realtime_processing_events": opts.speculation,
        },
    }
    if opts.custom_vocabulary:
        config["realtime_processing"] = {
            "custom_vocabulary": True,
            "custom_vocabulary_config": {"vocabulary": opts.custom_vocabulary},
        }
    return config


class STTv2(stt.STT):
    def __init__(
        self,
        *,
        model: V2Models | str = "solaria-3",
        languages: list[str] | None = None,
        code_switching: bool = False,
        region: V2Regions = "eu1",
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        eager_eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_end_of_turn_silence_ms: NotGivenOr[int] = NOT_GIVEN,
        silence_ms: NotGivenOr[int] = NOT_GIVEN,
        stop_s: NotGivenOr[float] = NOT_GIVEN,
        reeval_ms: NotGivenOr[int] = NOT_GIVEN,
        window_s: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        speculation: bool = False,
        speculation_threshold: NotGivenOr[float] = NOT_GIVEN,
        custom_vocabulary: NotGivenOr[list[str]] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Gladia STT on the regional realtime API, which ends the user's turn.

        Use it with ``turn_detection="stt"`` so the session takes Gladia's end of turn.

        Args:
            model: The model, optionally with a latency profile (``"solaria-3-fast"``).
                Defaults to ``"solaria-3"``.
            languages: The languages that may be spoken (ISO 639-1). Empty or None detects
                the language.
            code_switching: Detect the language as the conversation goes, among
                ``languages``. Defaults to False.
            region: The region of the account, ``"eu1"`` or ``"us1"``. Defaults to ``"eu1"``.
            eot_threshold: End-of-turn probability above which the turn ends, between 0
                and 1. Defaults to Gladia's (0.9).
            eager_eot_threshold: Probability from which an early end of turn is reported, so
                the reply starts before the turn is confirmed (preemptive generation). At
                most ``eot_threshold``. Off by default.
            min_end_of_turn_silence_ms: Silence that must follow a turn judged complete
                before its end is reported: higher cuts fewer users mid-sentence. Defaults
                to Gladia's (0).
            silence_ms: Silence, in milliseconds, before a pause is evaluated. Defaults to
                Gladia's (200).
            stop_s: Silence, in seconds, that ends the turn whatever the probability: the
                longest a user who is done waits for the reply. Defaults to Gladia's (1.5).
            reeval_ms: Interval, in milliseconds, between two evaluations of the same pause.
                Defaults to Gladia's (200).
            window_s: Audio, in seconds, an evaluation reads before the pause, at most 8.
                Defaults to Gladia's (8).
            vad_threshold: Speech probability, between 0 and 1, from which audio counts as
                speech. Defaults to Gladia's (0.5).
            speculation: Let Gladia judge whether the reply started on an early end of turn
                still answers the final transcript, rather than requiring the same words. A
                reply Gladia keeps is not regenerated, and its turn is recorded with the
                preflight's text. Needs ``eager_eot_threshold``. Off by default.
            speculation_threshold: Similarity, above 0 and at most 1, from which Gladia keeps
                the early reply. Defaults to Gladia's (0.85). Needs ``speculation``.
            custom_vocabulary: Words to bias the transcription toward, such as product names.
            api_key: The account's key for the regional API. Defaults to ``GLADIA_API_KEY``.
            base_url: The session endpoint, overriding ``region``.
            http_session: The HTTP session to use. Defaults to the agent's.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, offline_recognize=False
            )
        )
        gladia_api_key = api_key if is_given(api_key) else os.environ.get("GLADIA_API_KEY")
        if not gladia_api_key:
            raise ValueError("Gladia API key is required. Set GLADIA_API_KEY or pass api_key")
        if is_given(speculation_threshold) and not speculation:
            raise ValueError("speculation_threshold needs speculation=True")
        if is_given(eager_eot_threshold):
            effective_eot = eot_threshold if is_given(eot_threshold) else DEFAULT_EOT_THRESHOLD
            if eager_eot_threshold > effective_eot:
                raise ValueError(
                    f"eager_eot_threshold ({eager_eot_threshold}) must be at most "
                    f"eot_threshold ({effective_eot})"
                )

        self._api_key = gladia_api_key
        self._opts = STTOptions(
            model=model,
            languages=languages or [],
            code_switching=code_switching,
            base_url=base_url if is_given(base_url) else f"https://{region}.api.gladia.io/v2/live",
            eot_threshold=eot_threshold,
            eager_eot_threshold=eager_eot_threshold,
            min_end_of_turn_silence_ms=min_end_of_turn_silence_ms,
            silence_ms=silence_ms,
            stop_s=stop_s,
            reeval_ms=reeval_ms,
            window_s=window_s,
            vad_threshold=vad_threshold,
            speculation=speculation,
            speculation_threshold=speculation_threshold,
            custom_vocabulary=custom_vocabulary if is_given(custom_vocabulary) else [],
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStreamv2]()

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

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("STTv2 transcribes a stream: use stream()")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStreamv2:
        stream = SpeechStreamv2(stt=self, opts=self._opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream


class SpeechStreamv2(stt.SpeechStream):
    def __init__(self, *, stt: STTv2, opts: STTOptions, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._gladia = stt
        self._opts = opts
        self._language = LanguageCode(opts.languages[0] if opts.languages else "en")
        self._request_id = ""
        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report, duration=5.0
        )
        # the turn being spoken: the finals Gladia closed inside it, and the open partial
        self._finals: list[str] = []
        self._partial = ""
        self._in_turn = False
        self._closing_turn: asyncio.TimerHandle | None = None
        # a turn closed on its partial: the final Gladia still owes belongs to that turn
        self._late_final = False
        # with speculation, the preflight's text: Gladia's verdict on the final is about it
        self._speculated: str | None = None
        # the turn's final is in, its speculation verdict is not yet
        self._awaiting_verdict = False

    def _turn_text(self) -> str:
        return " ".join(t for t in (*self._finals, self._partial) if t).strip()

    def _send(self, event_type: stt.SpeechEventType, text: str | None = None) -> None:
        alternatives = (
            [stt.SpeechData(language=self._language, text=text, confidence=1.0)]
            if text is not None
            else []
        )
        self._event_ch.send_nowait(
            stt.SpeechEvent(type=event_type, request_id=self._request_id, alternatives=alternatives)
        )

    def _start_turn(self) -> None:
        # on the first words, not on speech_start: with STT turn detection a start of
        # speech interrupts the agent, and Gladia also reports one for a breath or a
        # noise it then transcribes as nothing
        if self._in_turn:
            return
        self._in_turn = True
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.START_OF_SPEECH,
                request_id=self._request_id,
                speech_start_time=time.time(),
            )
        )

    def _end_turn(self, text: str | None = None) -> None:
        if self._closing_turn is not None:
            self._closing_turn.cancel()
            self._closing_turn = None
        text = text or self._turn_text()
        self._finals, self._partial = [], ""
        self._speculated, self._awaiting_verdict = None, False
        if not self._in_turn:
            return
        self._in_turn = False
        now = time.time()
        if text:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[
                        stt.SpeechData(language=self._language, text=text, confidence=1.0)
                    ],
                    speech_end_time=now,
                )
            )
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH,
                request_id=self._request_id,
                speech_end_time=now,
            )
        )

    def _end_turn_unconfirmed(self) -> None:
        self._closing_turn = None
        # without a verdict the final is in, and only a missing final is still owed
        self._late_final = not self._awaiting_verdict
        self._end_turn()

    def _process_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        data = message.get("data") or {}
        if kind == "transcript":
            self._process_transcript(data)
        elif kind == "eager_end_of_turn":
            # with speculation Gladia sends the text it will compare the final to; it is
            # empty when Gladia had none, and then no verdict follows
            engine_text = (data.get("text") or "").strip() if self._opts.speculation else ""
            if self._in_turn and (text := engine_text or self._turn_text()):
                self._speculated = engine_text or None
                self._send(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, text)
        elif kind == "turn_resumed":
            self._speculated = None
            # the pipeline drops its preemptive reply once an interim moves past the
            # preflight transcript
            if self._in_turn and (text := self._turn_text()):
                self._send(stt.SpeechEventType.INTERIM_TRANSCRIPT, text)
        elif kind == "speculation":
            if self._awaiting_verdict:
                results = data.get("results") or [{}]
                keep = results[0].get("recommendation") == "keep"
                # the same words as the preflight are what makes the pipeline keep its reply
                self._end_turn(self._speculated if keep else None)
        elif kind == "end_of_turn":
            if self._closing_turn is not None:
                return  # already waiting for this turn's final
            if self._partial:
                self._closing_turn = asyncio.get_running_loop().call_later(
                    FINAL_WAIT_S, self._end_turn_unconfirmed
                )
            else:
                self._end_turn()
        elif kind == "error":
            logger.warning("gladia sent an error", extra={"lk.pii.data": data})

    def _process_transcript(self, data: dict[str, Any]) -> None:
        utterance = data.get("utterance") or {}
        text = (utterance.get("text") or "").strip()
        if language := utterance.get("language"):
            self._language = LanguageCode(language)
        if not text:
            return
        if data.get("is_final") and self._late_final:
            self._late_final = False
            return
        self._late_final = False
        self._start_turn()
        if not data.get("is_final"):
            self._partial = text
            self._send(stt.SpeechEventType.INTERIM_TRANSCRIPT, self._turn_text())
            return
        self._finals.append(text)
        self._partial = ""
        if self._closing_turn is not None:
            if self._speculated is not None:
                # Gladia's verdict on the early reply follows the final, within the same wait
                self._awaiting_verdict = True
                return
            self._end_turn()
            return
        self._send(stt.SpeechEventType.INTERIM_TRANSCRIPT, self._turn_text())

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    async def _init_session(self) -> str:
        try:
            async with self._gladia._ensure_session().post(
                self._opts.base_url,
                json=_session_config(self._opts),
                headers={"x-gladia-key": self._gladia._api_key},
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as res:
                body = await res.text()
                if res.status != 201:
                    # a 4xx is the request itself (a field, the key, the model): retrying
                    # cannot help
                    raise APIStatusError(
                        "gladia refused the session",
                        status_code=res.status,
                        body=body,
                        retryable=res.status >= 500 or res.status == 429,
                    )
                info: dict[str, Any] = json.loads(body)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIConnectionError(f"gladia unreachable ({type(e).__name__})") from None
        self._request_id = info.get("id", "")
        url: str = info["url"]
        return url

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        url = await self._init_session()
        try:
            return await self._gladia._ensure_session().ws_connect(url, heartbeat=20.0)
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"gladia websocket refused ({type(e).__name__})") from None

    async def _run(self) -> None:
        ws = await self._connect_ws()
        closing = False

        async def send_task() -> None:
            nonlocal closing
            chunker = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=SAMPLE_RATE * AUDIO_CHUNK_MS // 1000,
            )
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = (
                    chunker.write(data.data.tobytes())
                    if isinstance(data, rtc.AudioFrame)
                    else chunker.flush()
                )
                for frame in frames:
                    self._audio_duration_collector.push(frame.duration)
                    await ws.send_bytes(frame.data.tobytes())
            self._audio_duration_collector.flush()
            closing = True
            await ws.send_str(json.dumps({"type": "stop_recording"}))

        async def recv_task() -> None:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        self._process_message(json.loads(msg.data))
                    except Exception:
                        logger.exception("failed to process gladia message")
            if not closing:
                raise APIStatusError(
                    "gladia connection closed unexpectedly",
                    status_code=ws.close_code or -1,
                    retryable=True,
                )

        tasks = [asyncio.create_task(send_task()), asyncio.create_task(recv_task())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                task.result()
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            self._end_turn()
            await ws.close()
