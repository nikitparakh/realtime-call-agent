"""
Entry point for running voice_caller as a module.

Usage:
    python -m voice_caller --to "+15551234567" --purpose "Schedule a meeting"
    python -m voice_caller --server-only
"""

from .src.main import main

if __name__ == "__main__":
    main()
