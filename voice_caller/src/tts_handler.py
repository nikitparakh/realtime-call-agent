"""
Deepgram Text-to-Speech handler.
Provides streaming TTS with low-latency audio generation.
"""

import asyncio
import logging
from typing import Callable, Optional, Awaitable, AsyncIterator
from dataclasses import dataclass

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types import SpeakV1Text, SpeakV1Flush, SpeakV1Clear, SpeakV1Close

from .config import DeepgramConfig

logger = logging.getLogger(__name__)


@dataclass
class TTSState:
    """Tracks TTS state for a session."""
    is_connected: bool = False
    is_speaking: bool = False
    pending_text: str = ""


class TTSHandler:
    """Handles Deepgram TTS for real-time speech synthesis."""
    
    # Sentence boundary characters for flushing
    FLUSH_CHARS = {'.', '!', '?', ':', ';'}
    
    def __init__(self, config: DeepgramConfig):
        """
        Initialize the TTS handler.
        
        Args:
            config: Deepgram configuration
        """
        self.config = config
        self.client = AsyncDeepgramClient(api_key=config.api_key)
        self.connection = None
        self.state = TTSState()
        self._context_manager = None
        
        # Callbacks
        self._on_audio: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._on_complete: Optional[Callable[[], Awaitable[None]]] = None
        
        # Control flags
        self._cancel_event = asyncio.Event()
        self._listen_task: Optional[asyncio.Task] = None
    
    def on_audio(self, callback: Callable[[bytes], Awaitable[None]]) -> None:
        """Register callback for audio chunk events."""
        self._on_audio = callback
    
    def on_complete(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register callback for when TTS completes."""
        self._on_complete = callback
    
    async def connect(self) -> bool:
        """
        Establish connection to Deepgram TTS.
        
        Returns:
            True if connection successful
        """
        try:
            # Create context manager for connection
            self._context_manager = self.client.speak.v1.connect(
                model=self.config.tts_model,
                encoding=self.config.encoding,
                sample_rate=str(self.config.sample_rate),
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
            logger.info("Connected to Deepgram TTS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram TTS: {e}")
            return False
    
    async def send_text(self, text: str) -> None:
        """
        Send text to be synthesized.
        
        Args:
            text: Text to convert to speech
        """
        if not self.connection or not self.state.is_connected:
            logger.warning("TTS not connected, cannot send text")
            return
        
        if not text:
            return
        
        try:
            self.state.is_speaking = True
            await self.connection.send_text(SpeakV1Text(text=text))
            
            # Flush on sentence boundaries for natural pacing
            if text and text[-1] in self.FLUSH_CHARS:
                await self.connection.send_flush(SpeakV1Flush(type="Flush"))
                
        except Exception as e:
            logger.error(f"Error sending text to TTS: {e}")
    
    async def stream_text(self, text_iterator: AsyncIterator[str]) -> None:
        """
        Stream text tokens to TTS as they arrive from LLM.
        
        Args:
            text_iterator: Async iterator yielding text tokens
        """
        buffer = ""
        
        async for token in text_iterator:
            if self._cancel_event.is_set():
                logger.info("TTS streaming cancelled")
                break
            
            buffer += token
            
            # Send complete words (with trailing space) or sentence boundaries
            if token.endswith(' ') or (buffer and buffer[-1] in self.FLUSH_CHARS):
                await self.send_text(buffer)
                buffer = ""
        
        # Send any remaining text
        if buffer and not self._cancel_event.is_set():
            await self.send_text(buffer)
            await self.flush()
    
    async def flush(self) -> None:
        """Flush pending audio generation."""
        if self.connection and self.state.is_connected:
            try:
                await self.connection.send_flush(SpeakV1Flush(type="Flush"))
            except Exception as e:
                logger.error(f"Error flushing TTS: {e}")
    
    async def cancel(self) -> None:
        """Cancel ongoing TTS generation (for barge-in)."""
        logger.info("Cancelling TTS")
        self._cancel_event.set()
        self.state.is_speaking = False
        
        # Clear pending TTS
        if self.connection and self.state.is_connected:
            try:
                await self.connection.send_clear(SpeakV1Clear(type="Clear"))
            except Exception as e:
                logger.error(f"Error clearing TTS: {e}")
    
    def reset_cancel(self) -> None:
        """Reset the cancel flag for new generation."""
        self._cancel_event.clear()
    
    async def close(self) -> None:
        """Close the TTS connection."""
        if self.connection:
            try:
                await self.connection.send_close(SpeakV1Close(type="Close"))
            except Exception as e:
                logger.debug(f"Error sending close: {e}")
        
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing context manager: {e}")
            finally:
                self.state.is_connected = False
                self.connection = None
                self._context_manager = None
    
    def _handle_open(self, _) -> None:
        """Handle connection open event."""
        logger.debug("TTS connection opened")
    
    def _handle_close(self, _) -> None:
        """Handle connection close event."""
        logger.debug("TTS connection closed")
        self.state.is_connected = False
    
    def _handle_message(self, message) -> None:
        """Handle incoming messages from Deepgram."""
        # Run async handler in event loop
        asyncio.create_task(self._process_message(message))
    
    async def _process_message(self, message) -> None:
        """Process incoming messages asynchronously."""
        try:
            # Audio data comes as bytes
            if isinstance(message, bytes):
                # Immediately forward audio - no buffering for low latency
                if self._on_audio and not self._cancel_event.is_set():
                    await self._on_audio(message)
            else:
                # Check for control messages
                msg_type = getattr(message, "type", None)
                if msg_type == "Flushed":
                    logger.debug("TTS flushed")
                    self.state.is_speaking = False
                    if self._on_complete:
                        await self._on_complete()
                        
        except Exception as e:
            logger.error(f"Error processing TTS message: {e}")
    
    def _handle_error(self, error) -> None:
        """Handle error event."""
        logger.error(f"TTS error: {error}")
