"""
Audio format conversion utilities.
Handles conversion between different audio formats used by Telnyx and Deepgram.
"""

import base64
import audioop


def decode_mulaw(data: bytes) -> bytes:
    """Decode mu-law encoded audio to linear PCM."""
    return audioop.ulaw2lin(data, 2)


def encode_mulaw(data: bytes) -> bytes:
    """Encode linear PCM audio to mu-law."""
    return audioop.lin2ulaw(data, 2)


def base64_decode(data: str) -> bytes:
    """Decode base64 encoded audio data."""
    return base64.b64decode(data)


def base64_encode(data: bytes) -> str:
    """Encode audio data to base64."""
    return base64.b64encode(data).decode("utf-8")


def resample_audio(data: bytes, from_rate: int, to_rate: int, width: int = 2) -> bytes:
    """
    Resample audio data from one sample rate to another.
    
    Args:
        data: Raw audio bytes
        from_rate: Source sample rate
        to_rate: Target sample rate
        width: Sample width in bytes (2 for 16-bit audio)
    
    Returns:
        Resampled audio bytes
    """
    if from_rate == to_rate:
        return data
    
    converted, _ = audioop.ratecv(data, width, 1, from_rate, to_rate, None)
    return converted

