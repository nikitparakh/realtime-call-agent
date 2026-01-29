"""
Telnyx call management.
Handles outbound call initiation and call state management.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from telnyx import Telnyx

from .config import TelnyxConfig

logger = logging.getLogger(__name__)


@dataclass
class CallState:
    """Tracks the state of an active call."""
    call_control_id: str
    call_leg_id: str
    to_number: str
    from_number: str
    status: str = "initiated"
    stream_id: Optional[str] = None


class CallManager:
    """Manages Telnyx outbound calls."""
    
    def __init__(self, config: TelnyxConfig, public_ws_url: str):
        """
        Initialize the call manager.
        
        Args:
            config: Telnyx configuration
            public_ws_url: Public WebSocket URL for media streaming
        """
        self.config = config
        self.public_ws_url = public_ws_url
        self.client = Telnyx(api_key=config.api_key)
        self.active_calls: dict[str, CallState] = {}
    
    async def initiate_call(
        self,
        to_number: str,
        from_number: Optional[str] = None,
    ) -> CallState:
        """
        Initiate an outbound call with streaming enabled.
        
        Args:
            to_number: Destination phone number (E.164 format)
            from_number: Caller ID (defaults to configured number)
        
        Returns:
            CallState object tracking the call
        """
        from_number = from_number or self.config.phone_number
        
        logger.info(f"Initiating call to {to_number} from {from_number}")
        
        # Create the outbound call with streaming enabled from the start
        response = self.client.calls.dial(
            connection_id=self.config.connection_id,
            to=to_number,
            from_=from_number,
            answering_machine_detection="detect",
            answering_machine_detection_config={
                "after_greeting_silence_millis": 800,
                "between_words_silence_millis": 50,
                "greeting_duration_millis": 3500,
                "initial_silence_millis": 3500,
                "maximum_number_of_words": 5,
                "maximum_word_length_millis": 3500,
                "silence_threshold": 256,
                "total_analysis_time_millis": 5000,
            },
            # Enable bidirectional streaming from call start
            stream_url=self.public_ws_url,
            stream_track="both_tracks",
            # CRITICAL: Enable bidirectional mode to send audio back to caller
            stream_bidirectional_mode="rtp",
            stream_bidirectional_codec="PCMU",  # Match Deepgram's mulaw 8kHz
            webhook_url_method="POST",
        )
        
        # Response has data wrapper
        call_data = response.data
        call_control_id = call_data.call_control_id
        call_leg_id = call_data.call_leg_id
        
        call_state = CallState(
            call_control_id=call_control_id,
            call_leg_id=call_leg_id,
            to_number=to_number,
            from_number=from_number,
        )
        
        self.active_calls[call_control_id] = call_state
        logger.info(f"Call initiated with control_id: {call_control_id}")
        
        return call_state
    
    async def start_media_streaming(self, call_control_id: str) -> None:
        """
        Start media streaming for an answered call.
        
        Args:
            call_control_id: The call control ID
        """
        call_state = self.active_calls.get(call_control_id)
        if not call_state:
            logger.error(f"No active call found for {call_control_id}")
            return
        
        logger.info(f"Starting media stream for call {call_control_id}")
        
        # Start streaming to our WebSocket server
        self.client.calls.actions.start_streaming(
            call_control_id=call_control_id,
            stream_url=self.public_ws_url,
            stream_track="both_tracks",
            # Enable bidirectional mode to send audio back
            stream_bidirectional_mode="rtp",
            stream_bidirectional_codec="PCMU",
        )
        
        call_state.status = "streaming"
        logger.info(f"Media streaming started for call {call_control_id}")
    
    async def hangup(self, call_control_id: str) -> None:
        """
        Hang up a call.
        
        Args:
            call_control_id: The call control ID
        """
        logger.info(f"Hanging up call {call_control_id}")
        
        try:
            self.client.calls.actions.hangup(call_control_id=call_control_id)
        except Exception as e:
            logger.error(f"Error hanging up call: {e}")
        
        if call_control_id in self.active_calls:
            self.active_calls[call_control_id].status = "ended"
            del self.active_calls[call_control_id]
    
    def handle_webhook_event(self, event_type: str, payload: dict) -> Optional[str]:
        """
        Handle incoming Telnyx webhook events.
        
        Args:
            event_type: The type of event
            payload: Event payload
        
        Returns:
            Call control ID if relevant
        """
        call_control_id = payload.get("call_control_id")
        
        if event_type == "call.answered":
            logger.info(f"Call {call_control_id} answered")
            if call_control_id in self.active_calls:
                self.active_calls[call_control_id].status = "answered"
            return call_control_id
        
        elif event_type == "call.hangup":
            logger.info(f"Call {call_control_id} hung up")
            if call_control_id in self.active_calls:
                del self.active_calls[call_control_id]
            return call_control_id
        
        elif event_type == "streaming.started":
            stream_id = payload.get("stream_id")
            logger.info(f"Streaming started for call {call_control_id}, stream_id: {stream_id}")
            if call_control_id in self.active_calls:
                self.active_calls[call_control_id].stream_id = stream_id
            return call_control_id
        
        elif event_type == "call.machine.detection.ended":
            result = payload.get("result")
            logger.info(f"AMD result for {call_control_id}: {result}")
            return call_control_id
        
        return None
