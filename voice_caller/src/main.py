#!/usr/bin/env python3
"""
Voice Caller - Outbound calling with AI conversation.

Makes outbound phone calls using Telnyx, with real-time speech recognition
and synthesis via Deepgram, and conversational AI via Amazon Bedrock.

Usage:
    python main.py --to "+15551234567" --purpose "Schedule a meeting"
    python main.py --to "+15551234567" --server-only  # Just run the server
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import replace

import uvicorn

from .config import load_config
from .call_manager import CallManager
from .websocket_server import app, init_session_manager
from .llm_handler import LLMHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make outbound AI-powered phone calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Make a call with a specific purpose
    python main.py --to "+15551234567" --purpose "Remind about appointment tomorrow"
    
    # Make a call with custom voice
    python main.py --to "+15551234567" --voice "aura-2-zeus-en"
    
    # Just start the server (for testing with manual calls)
    python main.py --server-only
    
    # Run with debug logging
    python main.py --to "+15551234567" --debug
        """,
    )
    
    parser.add_argument(
        "--to",
        type=str,
        help="Destination phone number in E.164 format (e.g., +15551234567)",
    )
    
    parser.add_argument(
        "--from",
        dest="from_number",
        type=str,
        help="Caller ID number (defaults to configured TELNYX_PHONE_NUMBER)",
    )
    
    parser.add_argument(
        "--purpose",
        type=str,
        default="",
        help="Purpose of the call (helps guide the AI conversation)",
    )
    
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt for the AI (overrides default)",
    )
    
    parser.add_argument(
        "--voice",
        type=str,
        default="aura-2-thalia-en",
        help="Deepgram TTS voice model (default: aura-2-thalia-en)",
    )
    
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Only start the WebSocket server without making a call",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        help="Server host (overrides SERVER_HOST env var)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (overrides SERVER_PORT env var)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.server_only and not args.to:
        parser.error("--to is required unless using --server-only")
    
    return args


async def make_call(call_manager: CallManager, to_number: str, from_number: str = None) -> None:
    """
    Initiate an outbound call.
    
    Args:
        call_manager: The call manager instance
        to_number: Destination phone number
        from_number: Caller ID (optional)
    """
    logger.info(f"Initiating call to {to_number}")
    
    try:
        call_state = await call_manager.initiate_call(
            to_number=to_number,
            from_number=from_number,
        )
        logger.info(f"Call initiated: {call_state.call_control_id}")
        return call_state
        
    except Exception as e:
        logger.error(f"Failed to initiate call: {e}")
        raise


async def run_server_and_call(args: argparse.Namespace) -> None:
    """Run the server and optionally make a call."""
    
    # Load configuration
    config = load_config()
    
    # Override config from args
    host = args.host or config.server.host
    port = args.port or config.server.port
    
    # Update TTS model if specified
    if args.voice:
        config = replace(
            config,
            deepgram=replace(config.deepgram, tts_model=args.voice),
        )
    
    # Initialize call manager
    call_manager = CallManager(config.telnyx, config.server.public_ws_url)
    
    # Initialize session manager
    init_session_manager(config, call_manager)
    
    # Pre-generate system prompt and greeting BEFORE dialing for instant playback
    app.state.call_purpose = args.purpose or ""
    app.state.system_prompt = args.system_prompt
    app.state.pre_generated_greeting = None
    app.state.pre_generated_system_prompt = None
    
    if args.purpose and not args.server_only:
        logger.info(f"Pre-generating greeting and system prompt for: {args.purpose}")
        llm = LLMHandler(config.bedrock)
        system_prompt, greeting = await llm.initialize_for_call(args.purpose)
        app.state.pre_generated_greeting = greeting
        app.state.pre_generated_system_prompt = system_prompt
        logger.info(f"Pre-generated greeting: {greeting}")
        logger.info(f"System prompt ready ({len(system_prompt)} chars)")
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"WebSocket endpoint: ws://{host}:{port}/telnyx")
    logger.info(f"Public URL for Telnyx: {config.server.public_ws_url}")
    
    # Create server config
    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info" if not args.debug else "debug",
    )
    server = uvicorn.Server(server_config)
    
    # Start server in background task
    server_task = asyncio.create_task(server.serve())
    
    # Wait a moment for server to start
    await asyncio.sleep(1)
    
    # Make the call if not server-only mode
    if not args.server_only and args.to:
        try:
            await make_call(call_manager, args.to, args.from_number)
            logger.info("Call initiated. Waiting for call to complete...")
            logger.info("Press Ctrl+C to exit")
        except Exception as e:
            logger.error(f"Failed to make call: {e}")
    else:
        logger.info("Server running in server-only mode")
        logger.info("Press Ctrl+C to exit")
    
    # Wait for server to finish (or Ctrl+C)
    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Server shutdown requested")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("deepgram").setLevel(logging.DEBUG)
    
    try:
        asyncio.run(run_server_and_call(args))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

