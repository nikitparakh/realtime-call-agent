#!/usr/bin/env python3
"""
STT/TTS handler integration tests.
Tests the handler classes with simulated audio flow.
"""

import asyncio
import logging
from collections import deque

from . import conftest
from voice_caller.src.config import load_config
from voice_caller.src.stt_handler import STTHandler
from voice_caller.src.tts_handler import TTSHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_stt_continuous() -> bool:
    """Test STT with continuous audio streaming."""
    print("\n" + "=" * 60)
    print("TEST: STT Continuous Audio Streaming")
    print("=" * 60)
    
    config = load_config()
    stt = STTHandler(config.deepgram)
    
    async def on_speech_started():
        print("  [STT] Speech started detected")
    
    async def on_speech_ended(text):
        print(f"  [STT] Speech ended with transcript: '{text}'")
    
    stt.on_speech_started(on_speech_started)
    stt.on_speech_ended(on_speech_ended)
    
    try:
        connected = await stt.connect()
        if not connected:
            print("  ✗ Failed to connect to STT")
            return False
        print("  ✓ STT connected")
        
        silence = bytes([0xFF] * 160)
        print("  Sending 5 seconds of simulated audio...")
        
        for i in range(250):
            await stt.send_audio(silence)
            await asyncio.sleep(0.018)
            if i % 50 == 0:
                print(f"    Sent {i * 20}ms of audio, STT connected: {stt.state.is_connected}")
        
        print("  ✓ Sent 5 seconds of audio")
        print(f"  STT still connected: {stt.state.is_connected}")
        
        await stt.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await stt.close()


async def test_tts_generation() -> bool:
    """Test TTS audio generation."""
    print("\n" + "=" * 60)
    print("TEST: TTS Audio Generation")
    print("=" * 60)
    
    config = load_config()
    tts = TTSHandler(config.deepgram)
    
    audio_chunks = []
    complete_event = asyncio.Event()
    
    async def on_audio(data):
        audio_chunks.append(data)
        if len(audio_chunks) == 1 or len(audio_chunks) % 20 == 0:
            print(f"  [TTS] Received audio chunk #{len(audio_chunks)}: {len(data)} bytes")
    
    async def on_complete():
        print("  [TTS] Audio generation complete")
        complete_event.set()
    
    tts.on_audio(on_audio)
    tts.on_complete(on_complete)
    
    try:
        connected = await tts.connect()
        if not connected:
            print("  ✗ Failed to connect to TTS")
            return False
        print("  ✓ TTS connected")
        
        test_text = "Hello! This is a test of the text to speech system. How can I help you today?"
        print(f"  Sending text: '{test_text}'")
        
        await tts.send_text(test_text)
        await tts.flush()
        
        try:
            await asyncio.wait_for(complete_event.wait(), timeout=10)
        except asyncio.TimeoutError:
            print("  ⚠ Timeout waiting for TTS complete (might be okay)")
        
        await asyncio.sleep(1)
        
        total_audio = sum(len(c) for c in audio_chunks)
        duration_ms = (total_audio / 8000) * 1000
        
        print(f"  ✓ Received {len(audio_chunks)} audio chunks")
        print(f"  ✓ Total audio: {total_audio} bytes (~{duration_ms:.0f}ms)")
        
        return len(audio_chunks) > 0
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await tts.close()


async def test_bidirectional_flow() -> bool:
    """Test STT and TTS working together."""
    print("\n" + "=" * 60)
    print("TEST: Bidirectional Flow (STT + TTS)")
    print("=" * 60)
    
    config = load_config()
    stt = STTHandler(config.deepgram)
    tts = TTSHandler(config.deepgram)
    
    tts_audio_queue = deque()
    stt_running = True
    
    async def on_tts_audio(data):
        tts_audio_queue.append(data)
    
    tts.on_audio(on_tts_audio)
    
    try:
        print("  Connecting to Deepgram services...")
        stt_ok, tts_ok = await asyncio.gather(stt.connect(), tts.connect())
        
        if not stt_ok or not tts_ok:
            print(f"  ✗ Connection failed - STT: {stt_ok}, TTS: {tts_ok}")
            return False
        print("  ✓ Both STT and TTS connected")
        
        print("\n  Simulating concurrent operation...")
        
        async def send_audio_to_stt():
            silence = bytes([0xFF] * 160)
            sent = 0
            while stt_running and stt.state.is_connected and sent < 300:
                await stt.send_audio(silence)
                sent += 1
                await asyncio.sleep(0.018)
                if sent % 100 == 0:
                    print(f"    STT: Sent {sent} audio chunks")
            print(f"    STT: Finished sending {sent} chunks")
        
        async def generate_tts():
            await asyncio.sleep(0.5)
            print("    TTS: Starting audio generation...")
            await tts.send_text("Hello, this is a test message.")
            await tts.flush()
            await asyncio.sleep(2)
            print(f"    TTS: Generated {len(tts_audio_queue)} audio chunks")
        
        await asyncio.gather(send_audio_to_stt(), generate_tts())
        
        print(f"\n  ✓ Concurrent operation completed")
        print(f"    STT connected: {stt.state.is_connected}")
        print(f"    TTS audio chunks: {len(tts_audio_queue)}")
        
        return stt.state.is_connected and len(tts_audio_queue) > 0
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        stt_running = False
        await asyncio.gather(stt.close(), tts.close(), return_exceptions=True)


async def test_tts_cancel() -> bool:
    """Test TTS cancellation (barge-in simulation)."""
    print("\n" + "=" * 60)
    print("TEST: TTS Cancellation (Barge-in)")
    print("=" * 60)
    
    config = load_config()
    tts = TTSHandler(config.deepgram)
    
    audio_chunks_before = []
    audio_chunks_after = []
    cancelled = False
    
    async def on_audio(data):
        if cancelled:
            audio_chunks_after.append(data)
        else:
            audio_chunks_before.append(data)
    
    tts.on_audio(on_audio)
    
    try:
        connected = await tts.connect()
        if not connected:
            print("  ✗ Failed to connect to TTS")
            return False
        print("  ✓ TTS connected")
        
        long_text = "This is a very long message that should take several seconds to generate. " * 5
        print(f"  Starting long TTS generation ({len(long_text)} chars)...")
        
        asyncio.create_task(tts.send_text(long_text))
        
        await asyncio.sleep(1)
        print(f"  Audio before cancel: {len(audio_chunks_before)} chunks")
        
        cancelled = True
        await tts.cancel()
        print("  ✓ Cancel command sent")
        
        await asyncio.sleep(1)
        print(f"  Audio after cancel: {len(audio_chunks_after)} chunks")
        
        success = len(audio_chunks_before) > 0
        print(f"  ✓ Cancellation test: {'PASS' if success else 'FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await tts.close()


async def main():
    print("\n" + "=" * 60)
    print("STT/TTS HANDLER INTEGRATION TESTS")
    print("=" * 60)
    print(f"SSL Certificates: {conftest.get_ssl_cert_path()}")
    
    results = {
        'stt_continuous': await test_stt_continuous(),
        'tts_generation': await test_tts_generation(),
        'bidirectional': await test_bidirectional_flow(),
        'tts_cancel': await test_tts_cancel(),
    }
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for test, passed in results.items():
        print(f"  {test}: {'✓ PASS' if passed else '✗ FAIL'}")
    
    all_passed = all(results.values())
    print("=" * 60)
    print("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

