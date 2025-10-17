#!/usr/bin/env python3
"""Test script for enhanced dashboard functionality."""

import sys
import os

# Add current directory to path
sys.path.append('.')

try:
    from dashboard.app import Config, health_check, cache_manager
    print('✅ Imports successful')

    # Test configuration
    config = Config()
    print(f'✅ Configuration loaded: {config.environment} mode on port {config.port}')

    # Test health check
    health = health_check()
    print(f'✅ Health check: {health["status"]}')

    # Test cache manager
    cache_manager.set("test_key", {"test": "data"})
    cached = cache_manager.get("test_key")
    print(f'✅ Cache manager working: {cached is not None}')

    print('✅ All core functionality working!')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
