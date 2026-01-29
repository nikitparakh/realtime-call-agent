"""
Deepgram Speech-to-Text handler with Voice Activity Detection.
Provides real-time transcription with low-latency VAD.
"""

import asyncio
import logging
from typing import Callable, Optional, Awaitable
from dataclasses import dataclass, field

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen import (
    ListenV1Results,
    ListenV1SpeechStarted,
    ListenV1UtteranceEnd,
)

from .config import DeepgramConfig

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    """Result from speech transcription."""
    text: str
    is_final: bool
    speech_final: bool
    confidence: float = 0.0


@dataclass
class STTState:
    """Tracks STT state for a session."""
    is_connected: bool = False
    is_speaking: bool = False
    current_transcript: str = ""
    transcript_parts: list[str] = field(default_factory=list)


class STTHandler:
    """Handles Deepgram STT with VAD for real-time transcription."""
    
    def __init__(self, config: DeepgramConfig):
        """
        Initialize the STT handler.
        
        Args:
            config: Deepgram configuration
        """
        self.config = config
        self.client = AsyncDeepgramClient(api_key=config.api_key)
        self.connection = None
        self.state = STTState()
        self._context_manager = None
        
        # Callbacks
        self._on_transcript: Optional[Callable[[TranscriptResult], Awaitable[None]]] = None
        self._on_speech_started: Optional[Callable[[], Awaitable[None]]] = None
        self._on_speech_ended: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_utterance_end: Optional[Callable[[str], Awaitable[None]]] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._audio_count: int = 0
    
    def on_transcript(self, callback: Callable[[TranscriptResult], Awaitable[None]]) -> None:
        """Register callback for transcript events."""
        self._on_transcript = callback
    
    def on_speech_started(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register callback for when speech starts (for barge-in)."""
        self._on_speech_started = callback
    
    def on_speech_ended(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register callback for when speech ends with final transcript."""
        self._on_speech_ended = callback
    
    def on_utterance_end(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register callback for utterance end event."""
        self._on_utterance_end = callback
    
    async def connect(self) -> bool:
        """
        Establish connection to Deepgram STT.
        
        Returns:
            True if connection successful
        """
        try:
            # Create context manager for connection
            self._context_manager = self.client.listen.v1.connect(
                model=self.config.stt_model,
                encoding=self.config.encoding,
                sample_rate=str(self.config.sample_rate),
                channels="1",
                punctuate="true",
                interim_results="true",  # Get partial transcripts for barge-in
                endpointing=str(self.config.endpointing_ms),  # Short silence = end of speech
                utterance_end_ms=str(self.config.utterance_end_ms),  # Backup boundary
                vad_events="true",  # Get speech start/end events
                smart_format="true",
            )
            
            # Enter the context manager
            self.connection = await self._context_manager.__aenter__()
            
            # Register event handlers
            self.connection.on(EventType.OPEN, self._handle_open)
            self.connection.on(EventType.CLOSE, self._handle_close)
            self.connection.on(EventType.MESSAGE, self._handle_message)
            self.connection.on(EventType.ERROR, self._handle_error)
            
            # Start listening for events in background (runs forever)
            self._listen_task = asyncio.create_task(self.connection.start_listening())
            
            self.state.is_connected = True
            logger.info("Connected to Deepgram STT")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram STT: {e}")
            return False
    
    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio data to Deepgram for transcription.
        
        Args:
            audio_data: Raw audio bytes (mulaw 8kHz)
        """
        if not self.connection or not self.state.is_connected:
            return  # Silently skip if not connected
        
        try:
            await self.connection.send_media(audio_data)
            
            self._audio_count += 1
            if self._audio_count == 1 or self._audio_count % 500 == 0:
                logger.info(f"Sent {self._audio_count} audio chunks to STT")
                
        except Exception as e:
            logger.error(f"Error sending audio to STT: {e}")
            self.state.is_connected = False  # Mark as disconnected on error
    
    async def close(self) -> None:
        """Close the STT connection."""
        if self.connection:
            try:
                await self.connection.send_close_stream()
            except Exception as e:
                logger.debug(f"Error sending close stream: {e}")
            
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing context manager: {e}")
            finally:
                self.state.is_connected = False
                self.connection = None
                self._context_manager = None
    
    def reset_transcript(self) -> None:
        """Reset the current transcript state."""
        self.state.current_transcript = ""
        self.state.transcript_parts.clear()
    
    def get_full_transcript(self) -> str:
        """Get the complete transcript from all parts."""
        return " ".join(self.state.transcript_parts).strip()
    
    def _handle_open(self, _) -> None:
        """Handle connection open event."""
        logger.debug("STT connection opened")
    
    def _handle_close(self, _) -> None:
        """Handle connection close event."""
        logger.debug("STT connection closed")
        self.state.is_connected = False
    
    def _handle_message(self, message) -> None:
        """Handle incoming messages from Deepgram."""
        # Run async handler in event loop
        asyncio.create_task(self._process_message(message))
    
    async def _process_message(self, message) -> None:
        """Process incoming messages asynchronously."""
        try:
            # Check message type by looking at the 'type' attribute
            msg_type = getattr(message, "type", None)
            logger.debug(f"STT received message type: {msg_type}")
            
            if msg_type == "Results" or isinstance(message, ListenV1Results):
                await self._handle_transcript(message)
            elif msg_type == "SpeechStarted" or isinstance(message, ListenV1SpeechStarted):
                logger.info("STT: Speech started")
                await self._handle_speech_started()
            elif msg_type == "UtteranceEnd" or isinstance(message, ListenV1UtteranceEnd):
                logger.info("STT: Utterance end")
                await self._handle_utterance_end()
            else:
                logger.debug(f"STT: Unknown message type: {msg_type}")
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _handle_transcript(self, result) -> None:
        """Handle incoming transcript."""
        try:
            channel = getattr(result, "channel", None)
            if not channel:
                logger.debug("STT: No channel in result")
                return
            
            alternatives = getattr(channel, "alternatives", [])
            if not alternatives:
                logger.debug("STT: No alternatives in channel")
                return
            
            alt = alternatives[0]
            transcript = getattr(alt, "transcript", "")
            if not transcript:
                # Empty transcript is common during silence
                return
            
            is_final = getattr(result, "is_final", False) or False
            speech_final = getattr(result, "speech_final", False) or False
            confidence = getattr(alt, "confidence", 0.0) or 0.0
            
            # Log all non-empty transcripts
            logger.info(f"STT transcript: '{transcript}' (final={is_final}, speech_final={speech_final})")
            
            # Update state
            if is_final:
                self.state.transcript_parts.append(transcript)
            else:
                self.state.current_transcript = transcript
            
            # Create result
            transcript_result = TranscriptResult(
                text=transcript,
                is_final=is_final,
                speech_final=speech_final,
                confidence=confidence,
            )
            
            # Notify callback
            if self._on_transcript:
                await self._on_transcript(transcript_result)
            
            # If speech is final (VAD detected end), trigger callback
            if speech_final:
                full_transcript = self.get_full_transcript()
                if full_transcript and self._on_speech_ended:
                    await self._on_speech_ended(full_transcript)
                self.reset_transcript()
                
        except Exception as e:
            logger.error(f"Error handling transcript: {e}")
    
    async def _handle_speech_started(self) -> None:
        """Handle speech started event (for barge-in detection)."""
        logger.debug("Speech started detected")
        self.state.is_speaking = True
        
        if self._on_speech_started:
            await self._on_speech_started()
    
    async def _handle_utterance_end(self) -> None:
        """Handle utterance end event."""
        logger.debug("Utterance end detected")
        self.state.is_speaking = False
        
        full_transcript = self.get_full_transcript()
        if full_transcript and self._on_utterance_end:
            await self._on_utterance_end(full_transcript)
        self.reset_transcript()
    
    def _handle_error(self, error) -> None:
        """Handle error event."""
        logger.error(f"STT error: {error}")
