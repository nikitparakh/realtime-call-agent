"""
Configuration management for the voice calling module.
Loads settings from environment variables with validation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env from voice_caller dir, project root, or current directory
_src_dir = Path(__file__).parent
_voice_caller_dir = _src_dir.parent
_project_root = _voice_caller_dir.parent
load_dotenv(_voice_caller_dir / '.env')
load_dotenv(_project_root / '.env')
load_dotenv()


def _get_required_env(key: str) -> str:
    """Get a required environment variable or raise an error."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def _get_optional_env(key: str, default: str = "") -> str:
    """Get an optional environment variable with a default."""
    return os.getenv(key, default)


def _get_optional_int(key: str, default: int) -> int:
    """Get an optional integer environment variable with a default."""
    value = os.getenv(key)
    return int(value) if value else default


def _get_optional_float(key: str, default: float) -> float:
    """Get an optional float environment variable with a default."""
    value = os.getenv(key)
    return float(value) if value else default


@dataclass(frozen=True)
class TelnyxConfig:
    """Telnyx API configuration."""
    api_key: str
    connection_id: str
    phone_number: str


@dataclass(frozen=True)
class DeepgramConfig:
    """Deepgram API configuration."""
    api_key: str
    stt_model: str
    tts_model: str
    sample_rate: int = 8000
    encoding: str = "mulaw"
    endpointing_ms: int = 300
    utterance_end_ms: int = 1000


@dataclass(frozen=True)
class BedrockConfig:
    """AWS Bedrock configuration."""
    api_key: str
    region: str
    model_id: str
    max_tokens: int = 50
    temperature: float = 0.7


@dataclass(frozen=True)
class ServerConfig:
    """WebSocket server configuration."""
    host: str
    port: int
    public_ws_url: str


@dataclass(frozen=True)
class Config:
    """Main configuration container."""
    telnyx: TelnyxConfig
    deepgram: DeepgramConfig
    bedrock: BedrockConfig
    server: ServerConfig


def load_config() -> Config:
    """Load and validate all configuration from environment variables."""
    return Config(
        telnyx=TelnyxConfig(
            api_key=_get_required_env("TELNYX_API_KEY"),
            connection_id=_get_required_env("TELNYX_CONNECTION_ID"),
            phone_number=_get_required_env("TELNYX_PHONE_NUMBER"),
        ),
        deepgram=DeepgramConfig(
            api_key=_get_required_env("DEEPGRAM_API_KEY"),
            stt_model=_get_optional_env("DEEPGRAM_STT_MODEL", "nova-2"),
            tts_model=_get_optional_env("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"),
            endpointing_ms=_get_optional_int("DEEPGRAM_ENDPOINTING_MS", 300),
            utterance_end_ms=_get_optional_int("DEEPGRAM_UTTERANCE_END_MS", 1000),
        ),
        bedrock=BedrockConfig(
            api_key=_get_required_env("BEDROCK_API_KEY"),
            region=_get_optional_env("AWS_REGION", "us-east-1"),
            model_id=_get_optional_env("BEDROCK_MODEL_ID", "us.amazon.nova-pro-v1:0"),
            max_tokens=_get_optional_int("BEDROCK_MAX_TOKENS", 50),
            temperature=_get_optional_float("BEDROCK_TEMPERATURE", 0.7),
        ),
        server=ServerConfig(
            host=_get_optional_env("SERVER_HOST", "0.0.0.0"),
            port=_get_optional_int("SERVER_PORT", 8765),
            public_ws_url=_get_required_env("PUBLIC_WS_URL"),
        ),
    )
