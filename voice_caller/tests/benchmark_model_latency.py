#!/usr/bin/env python3
"""
End-to-end latency benchmark for Bedrock models.
Tests real conversation flows with streaming.
"""

import asyncio
import os
import re
import sys
import time

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# Models to test
MODELS = [
    ("us.amazon.nova-micro-v1:0", "Nova Micro"),
    ("us.amazon.nova-lite-v1:0", "Nova Lite"),
    ("us.amazon.nova-pro-v1:0", "Nova Pro"),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "Claude Haiku 4.5"),
    ("us.anthropic.claude-3-5-haiku-20241022-v1:0", "Claude 3.5 Haiku"),
]

# Test conversations
TESTS = [
    ("Hello?", "Quick greeting"),
    ("What are your plans tonight?", "Casual question"),
    ("Can you help me with something?", "Help request"),
    ("I'm thinking about dinner, maybe Italian.", "Follow-up context"),
]

SYSTEM_PROMPT = "You are a friendly voice assistant on a phone call. Keep responses brief and natural, under 30 words."


async def test_streaming(session, model_id, messages, system_prompt, api_key, region):
    """Test streaming response and return TTFT, total time, and response."""
    endpoint = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse-stream"
    
    payload = {
        "messages": messages,
        "system": [{"text": system_prompt}],
        "inferenceConfig": {"maxTokens": 100, "temperature": 0.7},
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    start = time.time()
    first_token = None
    response_text = ""
    text_pattern = re.compile(rb'"text":"((?:[^"\\]|\\.)*)"')
    
    try:
        async with session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status != 200:
                err = await resp.text()
                return -1, -1, f"Error {resp.status}: {err[:50]}"
            
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                for match in text_pattern.finditer(buffer):
                    try:
                        text = match.group(1).decode("utf-8")
                        if text:
                            if first_token is None:
                                first_token = time.time()
                            response_text += text
                    except (UnicodeDecodeError, UnicodeError):
                        pass
            
            total = (time.time() - start) * 1000
            ttft = (first_token - start) * 1000 if first_token else -1
            return ttft, total, response_text
            
    except Exception as e:
        return -1, -1, f"Exception: {e}"


async def run_benchmark():
    """Run the benchmark."""
    api_key = os.getenv("BEDROCK_API_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    if not api_key:
        print("Error: BEDROCK_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 80)
    print("BEDROCK MODEL LATENCY BENCHMARK - STREAMING")
    print("=" * 80)
    print()
    
    results = {name: {"ttft": [], "total": []} for _, name in MODELS}
    
    async with aiohttp.ClientSession() as session:
        for model_id, model_name in MODELS:
            print(f"\n### {model_name} ({model_id})")
            print("-" * 60)
            
            messages = []
            
            for user_input, test_name in TESTS:
                messages.append({"role": "user", "content": [{"text": user_input}]})
                
                ttft, total, response = await test_streaming(
                    session, model_id, messages, SYSTEM_PROMPT, api_key, region
                )
                
                if ttft > 0:
                    results[model_name]["ttft"].append(ttft)
                    results[model_name]["total"].append(total)
                    
                    if not response.startswith("Error") and not response.startswith("Exception"):
                        messages.append({"role": "assistant", "content": [{"text": response}]})
                    
                    response_short = response[:50] + "..." if len(response) > 50 else response
                    print(f"  {test_name:20} TTFT: {ttft:6.0f}ms  Total: {total:6.0f}ms")
                    print(f"    User: {user_input}")
                    print(f"    Bot:  {response_short}")
                else:
                    print(f"  {test_name:20} FAILED: {response[:60]}")
                
                await asyncio.sleep(0.3)
            
            if results[model_name]["ttft"]:
                avg_ttft = sum(results[model_name]["ttft"]) / len(results[model_name]["ttft"])
                avg_total = sum(results[model_name]["total"]) / len(results[model_name]["total"])
                print(f"\n  SUMMARY: Avg TTFT = {avg_ttft:.0f}ms, Avg Total = {avg_total:.0f}ms")
            
            await asyncio.sleep(1)
    
    # Final comparison
    print("\n")
    print("=" * 80)
    print("FINAL COMPARISON - SORTED BY TTFT (fastest first)")
    print("=" * 80)
    print()
    print(f"{'Model':<25} {'Avg TTFT':>12} {'Avg Total':>12} {'Min TTFT':>12} {'Max TTFT':>12}")
    print("-" * 80)
    
    summary = []
    for model_name, data in results.items():
        if data["ttft"]:
            summary.append({
                "name": model_name,
                "avg_ttft": sum(data["ttft"]) / len(data["ttft"]),
                "avg_total": sum(data["total"]) / len(data["total"]),
                "min_ttft": min(data["ttft"]),
                "max_ttft": max(data["ttft"]),
            })
    
    summary.sort(key=lambda x: x["avg_ttft"])
    
    for s in summary:
        print(f"{s['name']:<25} {s['avg_ttft']:>10.0f}ms {s['avg_total']:>10.0f}ms "
              f"{s['min_ttft']:>10.0f}ms {s['max_ttft']:>10.0f}ms")
    
    print()
    print("-" * 80)
    
    if len(summary) >= 2:
        fastest = summary[0]
        current = next((s for s in summary if "Haiku 4.5" in s["name"]), summary[-1])
        improvement = ((current["avg_ttft"] - fastest["avg_ttft"]) / current["avg_ttft"]) * 100
        
        print()
        print(f"🏆 FASTEST: {fastest['name']}")
        print(f"   Avg TTFT: {fastest['avg_ttft']:.0f}ms")
        print(f"   vs Claude Haiku 4.5: {improvement:.1f}% faster")
        print()
        print("RECOMMENDATION: Use", fastest['name'], "for lowest voice AI latency")


if __name__ == "__main__":
    asyncio.run(run_benchmark())

