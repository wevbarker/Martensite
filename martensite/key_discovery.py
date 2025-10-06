#!/usr/bin/env python3
"""
Key Discovery Module - Martensite
Follows best practices for API key discovery on Linux systems
Discovery order: Environment → Keyring → XDG Config
"""

import os
from pathlib import Path
from typing import Optional
import tomllib

# Standard environment variable names per provider documentation
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GEMINI_API_KEY",  # Fallback for Google
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
}


def get_api_key(provider: str) -> Optional[str]:
    """
    Discover API key for a provider following best practices.

    Discovery order:
    1. Environment variable (canonical name)
    2. OS keyring via Secret Service (if available)
    3. XDG config file (~/.config/llm-keys/config.toml)

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic', 'google')

    Returns:
        API key string or None if not found
    """
    provider = provider.lower()

    # 1. Check environment variable
    env_var = PROVIDER_ENV_VARS.get(provider)
    if env_var:
        key = os.getenv(env_var)
        if key:
            return key

    # Special case: Google/Gemini fallback
    if provider == "google":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            return gemini_key

    # 2. Try OS keyring (Secret Service)
    try:
        import keyring
        service_name = f"llm/{provider}"
        key = keyring.get_password(service_name, "default")
        if key:
            return key
    except ImportError:
        pass  # keyring not installed
    except Exception:
        pass  # keyring backend not available

    # 3. Check XDG config file
    xdg_config_home = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    config_file = xdg_config_home / "llm-keys" / "config.toml"

    if config_file.exists():
        try:
            with open(config_file, 'rb') as f:
                config_data = tomllib.load(f)
                provider_config = config_data.get(provider, {})
                key = provider_config.get("api_key")
                if key:
                    return key
        except Exception:
            pass  # Config file malformed or unreadable

    return None


def check_provider_availability(provider: str) -> tuple[bool, str]:
    """
    Check if API key is available for a provider and report source.

    Returns:
        (available: bool, source: str) where source is one of:
        'environment', 'keyring', 'xdg_config', 'not_found'
    """
    provider = provider.lower()

    # Check environment
    env_var = PROVIDER_ENV_VARS.get(provider)
    if env_var and os.getenv(env_var):
        return True, "environment"

    if provider == "google" and os.getenv("GEMINI_API_KEY"):
        return True, "environment"

    # Check keyring
    try:
        import keyring
        key = keyring.get_password(f"llm/{provider}", "default")
        if key:
            return True, "keyring"
    except Exception:
        pass

    # Check XDG config
    xdg_config_home = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    config_file = xdg_config_home / "llm-keys" / "config.toml"

    if config_file.exists():
        try:
            with open(config_file, 'rb') as f:
                config_data = tomllib.load(f)
                if provider in config_data and config_data[provider].get("api_key"):
                    return True, "xdg_config"
        except Exception:
            pass

    return False, "not_found"


def get_available_providers() -> dict[str, str]:
    """
    Get all available providers and their key sources.

    Returns:
        Dictionary mapping provider name to source location
    """
    available = {}
    for provider in PROVIDER_ENV_VARS.keys():
        is_available, source = check_provider_availability(provider)
        if is_available:
            available[provider] = source
    return available


if __name__ == "__main__":
    # Simple diagnostic
    print("Martensite Key Discovery Diagnostic")
    print("=" * 50)

    available = get_available_providers()
    if available:
        print("\nAvailable API keys:")
        for provider, source in available.items():
            print(f"  ✓ {provider:12} (from {source})")
    else:
        print("\n⚠ No API keys found")

    print("\nKey discovery order:")
    print("  1. Environment variables")
    print("  2. OS keyring (Secret Service)")
    print("  3. XDG config (~/.config/llm-keys/config.toml)")
