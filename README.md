# Voice Caller

AI-powered outbound calling system that makes phone calls with real-time conversational AI.

**Stack:**
- **Telephony**: Telnyx Voice API
- **Speech-to-Text**: Deepgram Nova-2
- **Text-to-Speech**: Deepgram Aura-2
- **LLM**: Amazon Bedrock (Claude/Nova)

## Features

- 🎙️ Real-time speech recognition with low latency
- 🔊 Natural text-to-speech voice synthesis
- 🤖 Streaming LLM responses for fast replies
- 🛑 Barge-in detection (interrupt the bot mid-speech)
- 📞 Outbound calling with customizable purpose
- ⚡ Pre-generated greetings for instant playback

## Project Structure

```
TelnyxDeepgram/
├── requirements.txt
├── env.example
└── voice_caller/
    ├── __init__.py
    ├── __main__.py
    ├── .env
    ├── src/
    │   ├── config.py           # Configuration management
    │   ├── main.py             # CLI entry point
    │   ├── call_manager.py     # Telnyx call management
    │   ├── websocket_server.py # WebSocket server for media streams
    │   ├── stt_handler.py      # Deepgram STT handler
    │   ├── tts_handler.py      # Deepgram TTS handler
    │   ├── llm_handler.py      # Bedrock LLM handler
    │   └── audio_utils.py      # Audio conversion utilities
    └── tests/
        ├── run_all.py
        ├── test_deepgram_connection.py
        ├── test_stt_tts_handlers.py
        ├── test_websocket_session.py
        └── benchmark_model_latency.py
```

## Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Copy the example env file and fill in your credentials:

```bash
cp env.example voice_caller/.env
```

**Required** environment variables:

| Variable | Description |
|----------|-------------|
| `TELNYX_API_KEY` | Your Telnyx API key |
| `TELNYX_CONNECTION_ID` | Telnyx Voice API connection ID |
| `TELNYX_PHONE_NUMBER` | Your Telnyx phone number (E.164 format) |
| `DEEPGRAM_API_KEY` | Your Deepgram API key |
| `BEDROCK_API_KEY` | AWS Bedrock API key |
| `PUBLIC_WS_URL` | Public WebSocket URL for Telnyx callbacks |

**Optional** environment variables (with defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `us.amazon.nova-pro-v1:0` | Bedrock model ID |
| `BEDROCK_MAX_TOKENS` | `50` | Max tokens per LLM response |
| `BEDROCK_TEMPERATURE` | `0.7` | LLM temperature (0.0-1.0) |
| `DEEPGRAM_STT_MODEL` | `nova-2` | Deepgram STT model |
| `DEEPGRAM_TTS_MODEL` | `aura-2-thalia-en` | Deepgram TTS voice |
| `DEEPGRAM_ENDPOINTING_MS` | `300` | Silence duration to end speech (ms) |
| `DEEPGRAM_UTTERANCE_END_MS` | `1000` | Backup utterance boundary (ms) |
| `SERVER_HOST` | `0.0.0.0` | WebSocket server host |
| `SERVER_PORT` | `8765` | WebSocket server port |

### 3. Expose WebSocket server

Telnyx needs to connect to your WebSocket server. Use ngrok or similar:

```bash
ngrok http 8765
```

Update `PUBLIC_WS_URL` in your `.env` with the ngrok URL + `/telnyx`:
```
PUBLIC_WS_URL=wss://your-subdomain.ngrok-free.dev/telnyx
```

## Usage

### Make an outbound call

```bash
# Activate venv
source venv/bin/activate

# Make a call with a purpose
python -m voice_caller --to "+15551234567" --purpose "Schedule a meeting for tomorrow"

# With custom voice
python -m voice_caller --to "+15551234567" --purpose "Follow up on order" --voice "aura-2-zeus-en"

# Debug mode
python -m voice_caller --to "+15551234567" --purpose "Test call" --debug
```

### Server-only mode

Run the WebSocket server without making a call (useful for testing):

```bash
python -m voice_caller --server-only
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--to` | Destination phone number (E.164 format) |
| `--from` | Caller ID (defaults to configured number) |
| `--purpose` | Purpose of the call (guides AI conversation) |
| `--system-prompt` | Custom system prompt (overrides default) |
| `--voice` | Deepgram TTS voice (default: aura-2-thalia-en) |
| `--server-only` | Only start server, don't make a call |
| `--host` | Server host (default: 0.0.0.0) |
| `--port` | Server port (default: 8765) |
| `--debug` | Enable debug logging |

## Running Tests

```bash
# Run all tests
python -m voice_caller.tests.run_all

# Run individual test suites
python -m voice_caller.tests.test_deepgram_connection
python -m voice_caller.tests.test_stt_tts_handlers
python -m voice_caller.tests.test_websocket_session

# Run LLM latency benchmark
python -m voice_caller.tests.benchmark_model_latency
```

## How It Works

1. **Call Initiation**: The CLI generates a greeting and system prompt based on the call purpose, then initiates an outbound call via Telnyx

2. **Media Streaming**: When the call connects, Telnyx streams audio to our WebSocket server in real-time (mulaw 8kHz)

3. **Speech Recognition**: Incoming audio is sent to Deepgram STT for real-time transcription with voice activity detection

4. **LLM Response**: When the user finishes speaking, the transcript is sent to Bedrock for a streaming response

5. **Text-to-Speech**: LLM tokens are streamed to Deepgram TTS, which generates audio sent back to the caller

6. **Barge-in**: If the user speaks while the bot is talking, TTS is cancelled and the new input is processed

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Caller    │◄───►│   Telnyx    │◄───►│  WebSocket  │
│   (Phone)   │     │  Voice API  │     │   Server    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
              ┌─────▼─────┐            ┌───────▼───────┐           ┌──────▼──────┐
              │  Deepgram │            │    Bedrock    │           │  Deepgram   │
              │    STT    │───────────►│     LLM       │──────────►│    TTS      │
              └───────────┘  transcript└───────────────┘  response └─────────────┘
```

## Troubleshooting

### Port already in use

```bash
lsof -ti:8765 | xargs kill -9
```

### Module not found errors

Make sure you've activated the virtual environment:
```bash
source venv/bin/activate
```

### WebSocket connection fails

- Check that ngrok is running and the URL is correct
- Verify `PUBLIC_WS_URL` ends with `/telnyx`
- Check Telnyx webhook configuration

## License

MIT

