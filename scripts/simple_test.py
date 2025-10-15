#!/usr/bin/env python3
"""
Simple test to debug the issue
"""

print("Script is starting...")

try:
    import sys
    print(f"Python version: {sys.version}")
    
    from pathlib import Path
    print(f"Current directory: {Path.cwd()}")
    
    # Test imports
    print("\nTesting imports...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from mcp_trader.data.api_manager import APIKeyManager
    print("✓ Imported APIKeyManager")
    
    # Test API manager
    print("\nTesting API Manager...")
    api_manager = APIKeyManager()
    credentials = api_manager.load_credentials()
    print("✓ Loaded credentials")
    print(credentials.get_status_report())
    
    # Test basic async
    import asyncio
    
    async def test_async():
        print("\n✓ Async function is working")
        return True
    
    print("\nTesting async...")
    result = asyncio.run(test_async())
    print(f"Async result: {result}")
    
    print("\n✅ All basic tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nScript finished.")

