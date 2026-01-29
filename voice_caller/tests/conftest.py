"""
Shared test configuration and utilities.
"""

import os
import sys
import certifi

# Add project root to path for imports
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_voice_caller_dir = os.path.dirname(_tests_dir)
_project_root = os.path.dirname(_voice_caller_dir)
sys.path.insert(0, _project_root)

# Set SSL certificates
os.environ['SSL_CERT_FILE'] = certifi.where()

from dotenv import load_dotenv
# Load .env from voice_caller dir first, then project root
load_dotenv(os.path.join(_voice_caller_dir, '.env'))
load_dotenv(os.path.join(_project_root, '.env'))


def get_ssl_cert_path() -> str:
    """Get the SSL certificate path."""
    return certifi.where()


def check_env_vars() -> dict[str, bool]:
    """Check which required environment variables are set."""
    required = ['DEEPGRAM_API_KEY', 'TELNYX_API_KEY', 'BEDROCK_API_KEY']
    return {var: bool(os.getenv(var)) for var in required}

