#!/usr/bin/env python3
"""
Deepgram connection tests.
Verifies STT and TTS WebSocket connections work correctly.
"""

import asyncio
import os

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types import SpeakV1Text, SpeakV1Flush

from . import conftest


async def test_stt_connection() -> bool:
    """Test Deepgram STT WebSocket connection."""
    print("\n=== Testing Deepgram STT ===")
    
    api_key = os.getenv('DEEPGRAM_API_KEY')
    if not api_key:
        print("ERROR: DEEPGRAM_API_KEY not set")
        return False
    
    print(f"API Key: {api_key[:10]}...")
    
    client = AsyncDeepgramClient(api_key=api_key)
    messages_received = []
    
    try:
        async with client.listen.v1.connect(
            model="nova-2",
            encoding="mulaw",
            sample_rate="8000",
            channels="1",
            punctuate="true",
            interim_results="true",
            endpointing="300",
            vad_events="true",
        ) as connection:
            print("✓ STT WebSocket connected!")
            
            def on_message(msg):
                msg_type = getattr(msg, 'type', type(msg).__name__)
                messages_received.append(msg_type)
                print(f"  Received: {msg_type}")
            
            connection.on(EventType.OPEN, lambda _: print("  Connection opened"))
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.CLOSE, lambda _: print("  Connection closed"))
            connection.on(EventType.ERROR, lambda err: print(f"  Error: {err}"))
            
            asyncio.create_task(connection.start_listening())
            
            # Send silence to keep connection alive
            silence = bytes([0xFF] * 320)
            for i in range(50):
                try:
                    await connection.send_media(silence)
                    await asyncio.sleep(0.02)
                except Exception as e:
                    print(f"  Send error at {i}: {e}")
                    break
            
            print("✓ Sent test audio data")
            print(f"✓ STT test passed! Received {len(messages_received)} messages")
            return True
            
    except Exception as e:
        print(f"✗ STT connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tts_connection() -> bool:
    """Test Deepgram TTS WebSocket connection."""
    print("\n=== Testing Deepgram TTS ===")
    
    api_key = os.getenv('DEEPGRAM_API_KEY')
    if not api_key:
        print("ERROR: DEEPGRAM_API_KEY not set")
        return False
    
    client = AsyncDeepgramClient(api_key=api_key)
    audio_chunks = []
    
    try:
        async with client.speak.v1.connect(
            model="aura-2-thalia-en",
            encoding="mulaw",
            sample_rate="8000",
        ) as connection:
            print("✓ TTS WebSocket connected!")
            
            def on_message(msg):
                if isinstance(msg, bytes):
                    audio_chunks.append(msg)
                    print(f"  Received audio: {len(msg)} bytes")
                else:
                    msg_type = getattr(msg, 'type', type(msg).__name__)
                    print(f"  Received: {msg_type}")
            
            connection.on(EventType.OPEN, lambda _: print("  Connection opened"))
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.CLOSE, lambda _: print("  Connection closed"))
            connection.on(EventType.ERROR, lambda err: print(f"  Error: {err}"))
            
            asyncio.create_task(connection.start_listening())
            await asyncio.sleep(0.1)
            
            test_text = "Hello, this is a test."
            await connection.send_text(SpeakV1Text(text=test_text))
            print(f"✓ Sent text: '{test_text}'")
            
            await connection.send_flush(SpeakV1Flush(type="Flush"))
            print("✓ Sent flush")
            
            await asyncio.sleep(3)
            
            total_audio = sum(len(c) for c in audio_chunks)
            print(f"✓ TTS test passed! Received {len(audio_chunks)} chunks, {total_audio} bytes total")
            return total_audio > 0
            
    except Exception as e:
        print(f"✗ TTS connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("Deepgram Connection Test")
    print("=" * 50)
    print(f"SSL Certificates: {conftest.get_ssl_cert_path()}")
    
    stt_ok = await test_stt_connection()
    tts_ok = await test_tts_connection()
    
    print("\n" + "=" * 50)
    print("Results:")
    print(f"  STT: {'✓ PASS' if stt_ok else '✗ FAIL'}")
    print(f"  TTS: {'✓ PASS' if tts_ok else '✗ FAIL'}")
    
    return stt_ok and tts_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

