"""
WebSocket server for orchestrating Telnyx media streams.
Bridges Telnyx audio with Deepgram STT/TTS and Bedrock LLM.
"""

import asyncio
import json
import logging
import os
import certifi
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

from .config import Config
from .stt_handler import STTHandler
from .tts_handler import TTSHandler
from .llm_handler import LLMHandler
from .audio_utils import base64_decode, base64_encode
from .call_manager import CallManager

os.environ['SSL_CERT_FILE'] = certifi.where()

logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Caller WebSocket Server")


@dataclass
class CallSession:
    """Manages a single call session with all components."""
    call_control_id: str
    stream_id: str
    config: Config
    stt: STTHandler
    tts: TTSHandler
    llm: LLMHandler
    websocket: Optional[WebSocket] = None
    is_active: bool = True
    is_bot_speaking: bool = False
    pending_response_task: Optional[asyncio.Task] = None
    audio_buffer: deque = field(default_factory=lambda: deque(maxlen=500))
    tts_audio_queue: deque = field(default_factory=lambda: deque(maxlen=1000))
    is_ready: bool = False
    barge_in_enabled: bool = False
    stt_enabled: bool = False
    greeting: str = "Hello, this is an AI assistant."
    # Counters initialized here instead of via hasattr checks
    tts_audio_count: int = 0
    tts_audio_sent: int = 0
    inbound_count: int = 0
    outbound_count: int = 0


class SessionManager:
    """Manages active call sessions."""
    
    def __init__(self, config: Config, call_manager: CallManager):
        self.config = config
        self.call_manager = call_manager
        self.sessions: dict[str, CallSession] = {}
    
    async def create_session(
        self,
        call_control_id: str,
        stream_id: str,
        websocket: WebSocket,
        purpose: Optional[str] = None,
    ) -> CallSession:
        """Create a new call session with all handlers."""
        logger.info(f"Creating session for call_control_id={call_control_id}, stream_id={stream_id}")
        
        stt = STTHandler(self.config.deepgram)
        tts = TTSHandler(self.config.deepgram)
        
        pre_greeting = getattr(app.state, 'pre_generated_greeting', None)
        pre_system_prompt = getattr(app.state, 'pre_generated_system_prompt', None)
        
        if pre_greeting and pre_system_prompt:
            logger.info("Using pre-generated greeting and system prompt (instant playback)")
            greeting = pre_greeting
            llm = LLMHandler(self.config.bedrock, system_prompt=pre_system_prompt, purpose=purpose)
            # Store greeting but don't add to history - Nova requires user message first
            llm._greeting_text = greeting
        elif purpose:
            logger.warning("No pre-generated content, generating now (adds latency)...")
            llm = LLMHandler(self.config.bedrock)
            _, greeting = await llm.initialize_for_call(purpose)
        else:
            llm = LLMHandler(self.config.bedrock)
            greeting = "Hello, this is an AI assistant."
        
        session = CallSession(
            call_control_id=call_control_id,
            stream_id=stream_id,
            config=self.config,
            stt=stt,
            tts=tts,
            llm=llm,
            websocket=websocket,
            greeting=greeting,
        )
        
        self.sessions[stream_id] = session
        await self._setup_callbacks(session)
        asyncio.create_task(self._connect_deepgram(session))
        
        logger.info(f"Session created for stream {stream_id}")
        return session
    
    async def _connect_deepgram(self, session: CallSession) -> None:
        """Connect to Deepgram services in background."""
        try:
            logger.info("Connecting to Deepgram STT and TTS...")
            
            stt_result, tts_result = await asyncio.gather(
                session.stt.connect(),
                session.tts.connect(),
                return_exceptions=True,
            )
            
            stt_ok = stt_result is True
            tts_ok = tts_result is True
            logger.info(f"STT connected: {stt_ok}, TTS connected: {tts_ok}")
            
            if stt_ok and tts_ok:
                session.is_ready = True
                
                buffered_count = len(session.audio_buffer)
                if buffered_count > 0:
                    logger.info(f"Discarding {buffered_count} buffered audio chunks (pre-greeting)")
                    session.audio_buffer.clear()
                
                await self._send_greeting(session, session.greeting)
            else:
                logger.error("Failed to connect to Deepgram services")
                
        except Exception as e:
            logger.error(f"Error connecting to Deepgram: {e}", exc_info=True)
    
    async def _send_greeting(self, session: CallSession, greeting: str) -> None:
        """Send an initial greeting to the caller."""
        try:
            logger.info(f"Sending greeting: {greeting}")
            session.is_bot_speaking = True
            session.stt_enabled = False
            session.barge_in_enabled = False
            
            await session.tts.send_text(greeting)
            await session.tts.flush()
            
            # Wait for TTS to generate audio
            wait_count = 0
            while wait_count < 50:
                if len(session.tts_audio_queue) > 10:
                    break
                await asyncio.sleep(0.02)
                wait_count += 1
            
            logger.info(f"Greeting audio arriving (queue: {len(session.tts_audio_queue)} chunks)")
            
            # Wait for queue to drain
            playback_chunks = 0
            while session.tts_audio_queue or playback_chunks < 50:
                await asyncio.sleep(0.02)
                playback_chunks += 1
                if playback_chunks > 500:
                    break
            
            await asyncio.sleep(0.5)  # Buffer for network latency
            
            logger.info("Greeting playback complete, enabling STT and barge-in")
            session.stt_enabled = True
            session.barge_in_enabled = True
        except Exception as e:
            logger.error(f"Error sending greeting: {e}")
        finally:
            session.is_bot_speaking = False
            session.stt_enabled = True
            session.barge_in_enabled = True
    
    async def _setup_callbacks(self, session: CallSession) -> None:
        """Set up callbacks between handlers."""
        
        async def on_speech_started():
            """Handle barge-in: user started speaking while bot is talking."""
            if session.is_bot_speaking and session.barge_in_enabled and session.tts_audio_sent > 10:
                logger.info(f"Barge-in detected (sent {session.tts_audio_sent} TTS chunks), cancelling TTS")
                await session.tts.cancel()
                session.is_bot_speaking = False
                session.tts_audio_sent = 0
                
                if session.pending_response_task and not session.pending_response_task.done():
                    session.pending_response_task.cancel()
            elif session.is_bot_speaking:
                logger.debug(f"Speech detected but not enough TTS sent ({session.tts_audio_sent} chunks)")
        
        async def on_user_finished(transcript: str):
            """Handle end of user speech: generate and speak response."""
            if not transcript.strip():
                return
            
            logger.info(f"User said: {transcript}")
            
            session.tts.reset_cancel()
            session.is_bot_speaking = True
            session.tts_audio_sent = 0
            
            session.pending_response_task = asyncio.create_task(
                self._stream_response(session, transcript)
            )
        
        async def on_tts_audio(audio_data: bytes):
            """Handle TTS audio: queue for sending to Telnyx."""
            if not session.is_active:
                return
            
            session.tts_audio_queue.append(audio_data)
            session.tts_audio_count += 1
            
            if session.tts_audio_count == 1 or session.tts_audio_count % 50 == 0:
                logger.info(f"TTS audio #{session.tts_audio_count}: {len(audio_data)} bytes (queued)")
        
        async def on_tts_complete():
            """Handle TTS completion."""
            session.is_bot_speaking = False
            logger.debug("TTS complete")
        
        session.stt.on_speech_started(on_speech_started)
        session.stt.on_speech_ended(on_user_finished)
        session.stt.on_utterance_end(on_user_finished)  # Same handler for both events
        session.tts.on_audio(on_tts_audio)
        session.tts.on_complete(on_tts_complete)
    
    async def _stream_response(self, session: CallSession, user_input: str) -> None:
        """Stream LLM response to TTS."""
        try:
            logger.info(f"Generating response for: {user_input[:50]}...")
            session.barge_in_enabled = True
            
            await session.tts.stream_text(session.llm.generate_response_stream(user_input))
            await session.tts.flush()
            
        except asyncio.CancelledError:
            logger.info("Response streaming cancelled")
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
        finally:
            session.is_bot_speaking = False
    
    async def handle_media(self, session: CallSession, payload: dict) -> None:
        """Handle incoming media from Telnyx."""
        try:
            media = payload.get("media", {})
            audio_base64 = media.get("payload")
            track = media.get("track", "inbound")
            
            if track != "inbound":
                return
            
            if audio_base64:
                audio_data = base64_decode(audio_base64)
                
                if session.is_ready and session.stt.state.is_connected and session.stt_enabled:
                    await session.stt.send_audio(audio_data)
                elif not session.stt_enabled:
                    pass  # Discard audio during greeting
                else:
                    session.audio_buffer.append(audio_data)
                    if len(session.audio_buffer) % 100 == 1:
                        logger.debug(f"Buffered {len(session.audio_buffer)} audio chunks")
                
        except Exception as e:
            logger.error(f"Error handling media: {e}")
    
    async def close_session(self, stream_id: str) -> None:
        """Close and clean up a session."""
        session = self.sessions.get(stream_id)
        if not session:
            return
        
        session.is_active = False
        
        if session.pending_response_task and not session.pending_response_task.done():
            session.pending_response_task.cancel()
        
        await asyncio.gather(
            session.stt.close(),
            session.tts.close(),
            return_exceptions=True,
        )
        
        del self.sessions[stream_id]
        logger.info(f"Closed session {stream_id}")


session_manager: Optional[SessionManager] = None


def init_session_manager(config: Config, call_manager: CallManager) -> None:
    """Initialize the global session manager."""
    global session_manager
    session_manager = SessionManager(config, call_manager)


@app.websocket("/telnyx")
async def telnyx_websocket(websocket: WebSocket):
    """WebSocket endpoint for Telnyx media streaming."""
    await websocket.accept()
    logger.info("Telnyx WebSocket connected!")
    
    stream_id: Optional[str] = None
    session: Optional[CallSession] = None
    
    async def drain_tts_queue():
        """Send queued TTS audio to Telnyx."""
        if not session or not session.tts_audio_queue:
            return
        
        sent = 0
        while session.tts_audio_queue and sent < 5:
            audio_data = session.tts_audio_queue.popleft()
            try:
                message = {
                    "event": "media",
                    "stream_id": session.stream_id,
                    "media": {"payload": base64_encode(audio_data)}
                }
                await websocket.send_json(message)
                sent += 1
                session.tts_audio_sent += 1
            except Exception as e:
                logger.error(f"Error sending TTS audio: {e}")
                break
    
    try:
        while True:
            await drain_tts_queue()
            
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            
            message = json.loads(data)
            event = message.get("event")
            
            if event == "connected":
                logger.info("Telnyx stream connected")
            
            elif event == "start":
                stream_id = message.get("stream_id")
                call_control_id = message.get("call_control_id", "")
                
                logger.info(f"Stream started: stream_id={stream_id}, call_control_id={call_control_id}")
                
                if session_manager:
                    purpose = getattr(app.state, 'call_purpose', None)
                    logger.info(f"Creating session with purpose: {purpose}")
                    
                    session = await session_manager.create_session(
                        call_control_id=call_control_id,
                        stream_id=stream_id,
                        websocket=websocket,
                        purpose=purpose,
                    )
            
            elif event == "media":
                if not session:
                    continue
                
                track = message.get("media", {}).get("track", "unknown")
                if track == "inbound":
                    session.inbound_count += 1
                    if session.inbound_count == 1 or session.inbound_count % 100 == 0:
                        logger.info(f"Inbound #{session.inbound_count}: stt={session.stt.state.is_connected}")
                else:
                    session.outbound_count += 1
                
                if session_manager:
                    await session_manager.handle_media(session, message)
            
            elif event == "stop":
                logger.info(f"Stream stopped: {stream_id}")
                if stream_id and session_manager:
                    await session_manager.close_session(stream_id)
                break
            
            elif event == "mark":
                logger.debug(f"Mark event: {message.get('name')}")
    
    except WebSocketDisconnect:
        logger.info("Telnyx WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if stream_id and session_manager:
            await session_manager.close_session(stream_id)


@app.post("/webhook")
async def telnyx_webhook(request: Request):
    """HTTP endpoint for Telnyx webhooks."""
    try:
        payload = await request.json()
        
        data = payload.get("data", {})
        event_type = data.get("event_type", "")
        event_payload = data.get("payload", {})
        
        logger.info(f"Webhook event: {event_type}")
        
        if session_manager:
            call_control_id = session_manager.call_manager.handle_webhook_event(
                event_type, event_payload
            )
            
            if event_type == "call.answered" and call_control_id:
                await session_manager.call_manager.start_media_streaming(call_control_id)
        
        return JSONResponse({"status": "ok"})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
