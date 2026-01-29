#!/usr/bin/env python3
"""
WebSocket session simulation tests.
Tests the WebSocket flow with a mock Telnyx connection.
"""

import asyncio
import json
import base64
import logging
from collections import deque

from . import conftest
from voice_caller.src.config import load_config
from voice_caller.src.stt_handler import STTHandler
from voice_caller.src.tts_handler import TTSHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockTelnyxWebSocket:
    """Simulates a Telnyx WebSocket connection."""
    
    def __init__(self):
        self.outbound_messages = deque()
        self.inbound_messages = deque()
        self.closed = False
    
    async def send_json(self, message):
        """Capture messages sent TO Telnyx."""
        self.outbound_messages.append(message)
        if len(self.outbound_messages) == 1 or len(self.outbound_messages) % 20 == 0:
            event = message.get('event', 'unknown')
            media_size = len(message.get('media', {}).get('payload', ''))
            logger.info(f"[Mock WS] Sent #{len(self.outbound_messages)}: event={event}, payload_size={media_size}")
    
    def add_inbound(self, message):
        """Add a message that would come FROM Telnyx."""
        self.inbound_messages.append(message)
    
    async def receive_text(self):
        """Receive a message FROM Telnyx."""
        while not self.closed:
            if self.inbound_messages:
                return json.dumps(self.inbound_messages.popleft())
            await asyncio.sleep(0.01)
        raise Exception("WebSocket closed")


async def test_session_creation_and_greeting() -> bool:
    """Test the session creation and greeting flow."""
    print("\n" + "=" * 60)
    print("TEST: Session Creation and Greeting Flow")
    print("=" * 60)
    
    config = load_config()
    stt = STTHandler(config.deepgram)
    tts = TTSHandler(config.deepgram)
    mock_ws = MockTelnyxWebSocket()
    
    audio_sent_to_telnyx = []
    
    async def on_tts_audio(data):
        audio_sent_to_telnyx.append(data)
        message = {
            "event": "media",
            "stream_id": "test-stream",
            "media": {"payload": base64.b64encode(data).decode('utf-8')}
        }
        await mock_ws.send_json(message)
    
    tts.on_audio(on_tts_audio)
    
    try:
        print("  1. Connecting to Deepgram...")
        stt_ok, tts_ok = await asyncio.gather(stt.connect(), tts.connect())
        print(f"     STT: {stt_ok}, TTS: {tts_ok}")
        
        if not stt_ok or not tts_ok:
            print("  ✗ Failed to connect")
            return False
        
        print("\n  2. Simulating buffered audio from Telnyx...")
        audio_buffer = deque()
        silence = bytes([0xFF] * 160)
        for _ in range(10):
            audio_buffer.append(silence)
        print(f"     Buffered {len(audio_buffer)} audio chunks")
        
        print("\n  3. Sending buffered audio to STT...")
        while audio_buffer:
            await stt.send_audio(audio_buffer.popleft())
        print("     Done")
        
        print("\n  4. Sending greeting via TTS...")
        greeting = "Hello! How can I help you today?"
        await tts.send_text(greeting)
        await tts.flush()
        await asyncio.sleep(3)
        
        print(f"     TTS generated {len(audio_sent_to_telnyx)} audio chunks")
        print(f"     Messages sent to mock Telnyx: {len(mock_ws.outbound_messages)}")
        
        print("\n  5. Simulating continuous audio FROM Telnyx...")
        for i in range(100):
            await stt.send_audio(silence)
            await asyncio.sleep(0.018)
            if i % 50 == 0:
                print(f"     Sent {i * 20}ms of audio to STT")
        
        print(f"     STT still connected: {stt.state.is_connected}")
        
        print("\n  Summary:")
        print(f"     TTS audio chunks generated: {len(audio_sent_to_telnyx)}")
        print(f"     Messages to Telnyx: {len(mock_ws.outbound_messages)}")
        print(f"     STT connected: {stt.state.is_connected}")
        
        return (
            len(audio_sent_to_telnyx) > 0 and
            len(mock_ws.outbound_messages) > 0 and
            stt.state.is_connected
        )
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        mock_ws.closed = True
        await asyncio.gather(stt.close(), tts.close(), return_exceptions=True)


async def test_concurrent_send_receive() -> bool:
    """Test concurrent sending and receiving."""
    print("\n" + "=" * 60)
    print("TEST: Concurrent Send/Receive Flow")
    print("=" * 60)
    
    config = load_config()
    stt = STTHandler(config.deepgram)
    tts = TTSHandler(config.deepgram)
    mock_ws = MockTelnyxWebSocket()
    
    stt_audio_count = 0
    tts_audio_count = 0
    telnyx_send_count = 0
    errors = []
    
    async def on_tts_audio(data):
        nonlocal tts_audio_count, telnyx_send_count
        tts_audio_count += 1
        
        try:
            message = {
                "event": "media",
                "stream_id": "test",
                "media": {"payload": base64.b64encode(data).decode('utf-8')}
            }
            await mock_ws.send_json(message)
            telnyx_send_count += 1
            await asyncio.sleep(0.01)
        except Exception as e:
            errors.append(f"TTS send error: {e}")
    
    tts.on_audio(on_tts_audio)
    
    try:
        print("  Connecting...")
        stt_ok, tts_ok = await asyncio.gather(stt.connect(), tts.connect())
        if not stt_ok or not tts_ok:
            print("  ✗ Connection failed")
            return False
        print("  ✓ Connected")
        
        print("\n  Running concurrent STT input + TTS output for 5 seconds...")
        
        async def stt_input():
            nonlocal stt_audio_count
            silence = bytes([0xFF] * 160)
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < 5:
                if not stt.state.is_connected:
                    errors.append("STT disconnected during test")
                    break
                await stt.send_audio(silence)
                stt_audio_count += 1
                await asyncio.sleep(0.018)
        
        async def tts_output():
            await asyncio.sleep(0.5)
            await tts.send_text("This is a test message to generate audio output.")
            await tts.flush()
            await asyncio.sleep(3)
        
        await asyncio.gather(stt_input(), tts_output())
        
        print(f"\n  Results:")
        print(f"    STT audio chunks sent: {stt_audio_count}")
        print(f"    TTS audio chunks received: {tts_audio_count}")
        print(f"    Messages sent to Telnyx: {telnyx_send_count}")
        print(f"    STT still connected: {stt.state.is_connected}")
        print(f"    Errors: {len(errors)}")
        
        for err in errors[:5]:
            print(f"      - {err}")
        
        return (
            stt_audio_count > 200 and
            tts_audio_count > 0 and
            stt.state.is_connected and
            len(errors) == 0
        )
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        mock_ws.closed = True
        await asyncio.gather(stt.close(), tts.close(), return_exceptions=True)


async def test_tts_rate_limiting() -> bool:
    """Test if TTS audio needs rate limiting when sending to Telnyx."""
    print("\n" + "=" * 60)
    print("TEST: TTS Rate Limiting for Telnyx")
    print("=" * 60)
    
    config = load_config()
    tts = TTSHandler(config.deepgram)
    
    audio_chunks = []
    timestamps = []
    
    async def on_audio(data):
        audio_chunks.append(data)
        timestamps.append(asyncio.get_event_loop().time())
    
    tts.on_audio(on_audio)
    
    try:
        connected = await tts.connect()
        if not connected:
            return False
        
        await tts.send_text("Hello, this is a test.")
        await tts.flush()
        await asyncio.sleep(3)
        
        if len(timestamps) < 2:
            print("  Not enough audio chunks")
            return False
        
        intervals = [
            (timestamps[i] - timestamps[i - 1]) * 1000
            for i in range(1, len(timestamps))
        ]
        
        avg_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)
        
        print(f"  Audio chunks: {len(audio_chunks)}")
        print(f"  Interval stats (ms):")
        print(f"    Average: {avg_interval:.2f}")
        print(f"    Min: {min_interval:.2f}")
        print(f"    Max: {max_interval:.2f}")
        print(f"  Expected real-time interval: ~40ms per chunk")
        
        if avg_interval < 10:
            print("\n  ⚠ WARNING: TTS sends much faster than real-time!")
            print("    Consider adding rate limiting.")
        
        return True
        
    finally:
        await tts.close()


async def main():
    print("\n" + "=" * 60)
    print("WEBSOCKET SESSION SIMULATION TESTS")
    print("(No actual Telnyx calls)")
    print("=" * 60)
    
    results = {
        'session_greeting': await test_session_creation_and_greeting(),
        'concurrent_flow': await test_concurrent_send_receive(),
        'tts_rate_limiting': await test_tts_rate_limiting(),
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

