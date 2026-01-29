"""
Amazon Bedrock Claude handler.
Provides streaming conversational AI responses.
"""

import asyncio
import logging
import re
import time
from typing import AsyncIterator, Optional
from dataclasses import dataclass, field

import aiohttp

from .config import BedrockConfig

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant on a phone call. Follow these guidelines:

1. Keep responses concise and natural - speak as you would in a real conversation
2. Use short sentences that are easy to speak and understand
3. Avoid lists, bullet points, or complex formatting - use flowing speech
4. Don't use special characters, emojis, or markdown
5. If you don't understand something, ask for clarification naturally
6. Be friendly, warm, and conversational
7. Acknowledge what the caller said before responding
8. End responses naturally without asking unnecessary follow-up questions

You're here to help the caller with their request."""


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationState:
    """Tracks the conversation state."""
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    purpose: Optional[str] = None


class LLMHandler:
    """Handles Bedrock Claude for conversational AI."""
    
    # Pattern for extracting text from Bedrock binary stream
    _TEXT_PATTERN = re.compile(rb'"text":"((?:[^"\\]|\\.)*)"')
    
    def __init__(self, config: BedrockConfig, system_prompt: Optional[str] = None, purpose: Optional[str] = None):
        self.config = config
        self.api_key = config.api_key
        self.region = config.region
        self.model_id = config.model_id
        
        self._base_url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model_id}"
        
        self.state = ConversationState(
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            purpose=purpose,
        )
        
        if purpose:
            self.state.system_prompt += f"\n\nCall purpose: {purpose}"
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        self.state.messages.append(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self.state.messages.append(Message(role="assistant", content=content))
    
    async def _make_request(
        self,
        messages: list[dict],
        system_text: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: int = 30,
        stream: bool = False,
    ) -> Optional[aiohttp.ClientResponse]:
        """
        Make a request to the Bedrock API.
        
        Returns the response object for streaming, or None on error.
        """
        endpoint = f"{self._base_url}/converse-stream" if stream else f"{self._base_url}/converse"
        
        # Use config defaults if not specified
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature
        
        payload = {
            "messages": messages,
            "inferenceConfig": {"maxTokens": tokens, "temperature": temp},
        }
        if system_text:
            payload["system"] = [{"text": system_text}]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        session = aiohttp.ClientSession()
        try:
            response = await session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            )
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Bedrock API error {response.status}: {error_text}")
                await session.close()
                return None
            
            # For streaming, return response (caller must close session)
            if stream:
                response._session = session  # Attach session for cleanup
                return response
            
            # For non-streaming, return response and let caller handle
            response._session = session
            return response
            
        except Exception as e:
            logger.error(f"Bedrock request error: {e}")
            await session.close()
            return None
    
    async def _extract_response_text(self, response: aiohttp.ClientResponse) -> Optional[str]:
        """Extract text content from a non-streaming Bedrock response."""
        try:
            result = await response.json()
            content = result.get("output", {}).get("message", {}).get("content", [])
            if content:
                return content[0].get("text", "")
        except Exception as e:
            logger.error(f"Error extracting response: {e}")
        finally:
            await response._session.close()
        return None
    
    async def generate_response_stream(self, user_input: str) -> AsyncIterator[str]:
        """
        Generate a streaming response to user input.
        
        Yields:
            Text tokens as they are generated
        """
        start_time = time.time()
        first_token_time = None
        
        # Handle first message - include greeting context in system prompt
        greeting = getattr(self, '_greeting_text', None)
        system_prompt = self.state.system_prompt
        if greeting and not self.state.messages:
            # First user message - add greeting context to system prompt
            system_prompt = f"{self.state.system_prompt}\n\nYou just said to the caller: \"{greeting}\"\nNow respond to their reply."
            # Clear the greeting so we don't add it again
            self._greeting_text = None
        
        self.add_user_message(user_input)
        
        messages = [
            {"role": msg.role, "content": [{"text": msg.content}]}
            for msg in self.state.messages
        ]
        
        logger.info(f"Generating response for: {user_input[:100]}...")
        
        response = await self._make_request(
            messages=messages,
            system_text=system_prompt,
            stream=True,
        )
        
        if not response:
            fallback = "I'm sorry, I'm having trouble responding right now."
            self.add_assistant_message(fallback)
            yield fallback
            return
        
        full_response = ""
        text_buffer = ""
        buffer = b""
        last_processed = 0
        
        try:
            async for chunk in response.content.iter_any():
                buffer += chunk
                
                for match in self._TEXT_PATTERN.finditer(buffer, last_processed):
                    try:
                        text_bytes = match.group(1)
                        text = text_bytes.decode('utf-8').encode('utf-8').decode('unicode_escape')
                        
                        if text:
                            if first_token_time is None:
                                first_token_time = time.time()
                                logger.info(f"First token latency: {(first_token_time - start_time) * 1000:.0f}ms")
                            
                            full_response += text
                            text_buffer += text
                            
                            # Yield at sentence boundaries for natural TTS
                            if text_buffer.rstrip().endswith(('.', '!', '?')):
                                yield text_buffer
                                text_buffer = ""
                            elif len(text_buffer) > 40 and text.endswith(' '):
                                yield text_buffer
                                text_buffer = ""
                        
                        last_processed = match.end()
                    except (UnicodeDecodeError, UnicodeError):
                        continue
            
            if text_buffer:
                yield text_buffer
            
            if full_response:
                self.add_assistant_message(full_response)
                logger.info(f"Response complete ({(time.time() - start_time) * 1000:.0f}ms): {full_response[:100]}...")
            else:
                fallback = "I'm sorry, could you repeat that?"
                self.add_assistant_message(fallback)
                yield fallback
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error: {e}")
            fallback = "I'm sorry, I'm having trouble connecting."
            self.add_assistant_message(fallback)
            yield fallback
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            fallback = "I'm sorry, I'm having trouble responding."
            self.add_assistant_message(fallback)
            yield fallback
        finally:
            if response and hasattr(response, '_session') and response._session:
                await response._session.close()
    
    async def generate_response(self, user_input: str) -> str:
        """Generate a complete response (non-streaming)."""
        response_parts = []
        async for token in self.generate_response_stream(user_input):
            response_parts.append(token)
        return "".join(response_parts)
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.state.messages.clear()
    
    def get_history(self) -> list[Message]:
        """Get the conversation history."""
        return self.state.messages.copy()
    
    async def generate_system_prompt(self, purpose: str) -> str:
        """Generate a tailored system prompt based on the call purpose."""
        meta_prompt = f"""You are creating a system prompt for a voice AI agent that will make a phone call.

The purpose of this call is: {purpose}

Generate a concise system prompt (max 200 words) that:
1. Defines the agent's role and goal for THIS specific call
2. Sets appropriate guardrails for professional conduct
3. Instructs the agent to be conversational and natural
4. Reminds the agent to keep responses short (suitable for voice)
5. Includes any relevant context for the call purpose

Output ONLY the system prompt text, nothing else. Do not include any meta-commentary."""

        messages = [{"role": "user", "content": [{"text": meta_prompt}]}]
        response = await self._make_request(messages=messages, max_tokens=500)
        
        if response:
            text = await self._extract_response_text(response)
            if text:
                logger.info(f"Generated system prompt: {text[:100]}...")
                return text
        
        return f"{DEFAULT_SYSTEM_PROMPT}\n\nCall purpose: {purpose}"
    
    async def generate_greeting(self, purpose: str) -> str:
        """Generate an appropriate opening greeting based on the call purpose."""
        meta_prompt = f"""Generate a brief, natural opening greeting for a phone call.

The purpose of this call is: {purpose}

Requirements:
- Keep it under 20 words
- Be friendly and professional
- Introduce yourself as an AI assistant
- Naturally lead into the call purpose
- Do NOT ask "how can I help you" - you know why you're calling

Output ONLY the greeting text, nothing else."""

        messages = [{"role": "user", "content": [{"text": meta_prompt}]}]
        response = await self._make_request(messages=messages, max_tokens=50, temperature=0.8, timeout=15)
        
        if response:
            text = await self._extract_response_text(response)
            if text:
                greeting = text.strip().strip('"\'')
                logger.info(f"Generated greeting: {greeting}")
                return greeting
        
        return f"Hello, this is an AI assistant calling about {purpose}."
    
    async def initialize_for_call(self, purpose: str) -> tuple[str, str]:
        """
        Initialize the LLM handler for a call by generating system prompt and greeting.
        
        Returns:
            Tuple of (system_prompt, greeting)
        """
        system_prompt, greeting = await asyncio.gather(
            self.generate_system_prompt(purpose),
            self.generate_greeting(purpose),
        )
        
        self.state.system_prompt = system_prompt
        self.state.purpose = purpose
        # Don't add greeting to history - Nova requires user message first
        # Store it so we can add it after first user message
        self._greeting_text = greeting
        
        return system_prompt, greeting
