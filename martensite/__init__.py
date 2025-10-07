"""
Martensite - Adversarial hardening for modern grantsmanship

A multi-LLM application review system for strengthening academic grant applications
through adversarial critique from diverse AI perspectives.

Active modules:
    - key_discovery: Secure API key management
    - martensite_handler: CLI backend (text extraction approach)

Experimental modules (not wired to CLI):
    - application_reviewer: Class-based implementation with native PDF support
"""

from .key_discovery import get_api_key, check_provider_availability, get_available_providers

__version__ = "0.1.0"
__all__ = [
    "get_api_key",
    "check_provider_availability",
    "get_available_providers",
]

# Experimental: ApplicationReviewer (not currently used by CLI)
# This class provides native PDF support for Claude/Gemini but is not yet integrated
try:
    from .application_reviewer import ApplicationReviewer, ReviewConfig, ReviewResult
    __all__.extend(["ApplicationReviewer", "ReviewConfig", "ReviewResult"])
except ImportError:
    pass
