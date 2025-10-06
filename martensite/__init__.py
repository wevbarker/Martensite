"""
Martensite - Adversarial hardening for modern grantsmanship

A multi-LLM application review system for strengthening academic grant applications
through adversarial critique from diverse AI perspectives.
"""

from .key_discovery import get_api_key, check_provider_availability, get_available_providers

__version__ = "0.1.0"
__all__ = [
    "get_api_key",
    "check_provider_availability",
    "get_available_providers",
]

# Optional imports - only load if dependencies are available
try:
    from .application_reviewer import ApplicationReviewer, ReviewConfig, ReviewResult
    __all__.extend(["ApplicationReviewer", "ReviewConfig", "ReviewResult"])
except ImportError:
    pass
