#!/usr/bin/env python3
"""
Run all test suites.
"""

import asyncio
import sys

from . import conftest
from .test_deepgram_connection import main as test_deepgram
from .test_stt_tts_handlers import main as test_handlers
from .test_websocket_session import main as test_websocket


async def main():
    print("\n" + "=" * 70)
    print("VOICE CALLER TEST SUITE")
    print("=" * 70)
    
    env_status = conftest.check_env_vars()
    print("\nEnvironment variables:")
    for var, is_set in env_status.items():
        print(f"  {var}: {'✓' if is_set else '✗ NOT SET'}")
    
    if not all(env_status.values()):
        print("\n⚠ Some environment variables are missing. Tests may fail.")
    
    results = {}
    
    print("\n" + "=" * 70)
    print("1. DEEPGRAM CONNECTION TESTS")
    print("=" * 70)
    results['deepgram_connection'] = await test_deepgram()
    
    print("\n" + "=" * 70)
    print("2. STT/TTS HANDLER TESTS")
    print("=" * 70)
    results['stt_tts_handlers'] = await test_handlers()
    
    print("\n" + "=" * 70)
    print("3. WEBSOCKET SESSION TESTS")
    print("=" * 70)
    results['websocket_session'] = await test_websocket()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    for suite, passed in results.items():
        print(f"  {suite}: {'✓ PASS' if passed else '✗ FAIL'}")
    
    all_passed = all(results.values())
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL TEST SUITES PASSED")
    else:
        print("✗ SOME TEST SUITES FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

