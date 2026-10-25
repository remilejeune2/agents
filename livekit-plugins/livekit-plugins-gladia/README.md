# Gladia plugin for LiveKit Agents

Support for speech-to-text with [Gladia](https://gladia.io/).

See [https://docs.livekit.io/agents/integrations/stt/gladia/](https://docs.livekit.io/agents/integrations/stt/gladia/) for more information.

## Installation

```bash
pip install livekit-plugins-gladia
```

## Pre-requisites

You'll need an API key from Gladia. It can be set as an environment variable: `GLADIA_API_KEY`

## Features

- Streaming speech-to-text
- Multi-language support
- Code-switching between languages
- Interim results (partial transcriptions)
- Voice activity detection with energy filtering
- Optional real-time translation
- Customizable audio parameters (sample rate, bit depth, channels, encoding)

## Example Usage

```python
from livekit.stt import STT
from livekit.plugins.gladia.stt import STT as GladiaSTT

# Basic initialization
stt = GladiaSTT(
    api_key="your-api-key-here",  # or use GLADIA_API_KEY env var
    interim_results=True
)

# With more options
stt = GladiaSTT(
    languages=["en", "fr"],                     # Specify languages or let Gladia auto-detect
    code_switching=True,                        # Allow switching between languages during recognition
    sample_rate=16000,                          # Audio sample rate in Hz
    bit_depth=16,                               # Audio bit depth
    channels=1,                                 # Number of audio channels
    region="eu-west"                            # Specify Region to use for the Gladia API
    encoding="wav/pcm",                         # Audio encoding format
    energy_filter=True,                         # Enable voice activity detection
    translation_enabled=True,
    translation_target_languages=["en"],
    translation_model="base",
    translation_match_original_utterances=True
    translation_context_adaptation= False,      # Enable context-aware translation
    translation_context= None,                  # Context input to guide translation
    translation_informal=False,                 # Use informal tone in translation
    pre_processing_audio_enhancer=False,        # Apply pre-processing to the audio stream to enhance the quality
    pre_processing_speech_threshold=0.6,        # Sensitivity for speech detection; closer to 1 = stricter, less background noise

    # Custom_vocabulary exemple
    custom_vocabulary=[
        "Westeros",
        {"value": "Stark"},
        {
            "value": "Night's Watch",
            "pronunciations": ["Nightz Watch"],
            "intensity": 0.4,
            "language": "en"
        }
    ],

    # Custom_spelling exemple
    custom_spelling={
        "Gorish": ["ghorish", "gaurish", "gaureish"],
        "Data Science": ["data-science", "data science"],
        ".": ["period", "full stop"],
        "SQL": ["sequel"]
    }
)

# Update options after initialization
stt.update_options(
    languages=["ja", "en"],
    translation_enabled=True,
    translation_target_languages=["fr"]
)
```

## Using with LiveKit Agents Framework

```python
from livekit.agents import Agent
from livekit.plugins.gladia.stt import STT as GladiaSTT

agent = Agent(
    stt=GladiaSTT(
        api_key="your-api-key-here",
        languages=["en"],
        translation_enabled=True,
        translation_target_languages=["es"]
    )
)

# Rest of your agent setup...
```

## End-of-turn detection on Gladia's regional API (`STTv2`)

`STTv2` uses Gladia's regional realtime API (`eu1`, `us1`), which decides when the user has
finished their turn from how they speak, not from silence alone. Let the session take that
decision with `turn_detection="stt"`:

```python
from livekit.agents import AgentSession
from livekit.plugins import gladia

session = AgentSession(
    stt=gladia.STTv2(
        model="solaria-3",           # or a latency profile: "trantoria-1-fast", "solaria-3-fast"…
        languages=["fr"],
        region="eu1",
        eager_eot_threshold=0.6,     # optional: start the reply before the turn is confirmed
    ),
    turn_handling={"turn_detection": "stt", "endpointing": {"min_delay": 0.0}},
    # llm=..., tts=..., vad=silero.VAD.load() for interruptions
)
```

The key is your regional account's key (`GLADIA_API_KEY` or `api_key`). Gladia's turn messages
map to the session's events: `eager_end_of_turn` to a preflight transcript (preemptive
generation), `turn_resumed` to an interim that drops it, and `end_of_turn` to one final transcript
for the whole turn followed by the end of speech. A pause inside a turn never produces a final, so
it never ends the turn.
