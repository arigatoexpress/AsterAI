#!/usr/bin/env python3
"""
Test script to verify data collection setup
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.api_manager import APIKeyManager
from scripts.collect_multi_source_crypto import MultiSourceCryptoCollector


async def test_setup():
    """Test basic setup"""
    print("Testing Ultimate Data Collection Setup...")
    print("=" * 60)
    
    # Test API manager
    print("\n1. Testing API Manager...")
    try:
        api_manager = APIKeyManager()
        credentials = api_manager.load_credentials()
        print(credentials.get_status_report())
    except Exception as e:
        print(f"❌ API Manager Error: {e}")
        return False
    
    # Test crypto collector initialization
    print("\n2. Testing Crypto Collector...")
    try:
        collector = MultiSourceCryptoCollector(
            output_dir="data/test_collection"
        )
        await collector.initialize()
        print("✅ Crypto collector initialized successfully")
        
        # Test collecting one asset
        print("\n3. Testing data collection for BTC...")
        result = await collector.collect_asset_data("BTC")
        
        if result and result.get('metadata', {}).get('data_points', 0) > 0:
            print(f"✅ Successfully collected {result['metadata']['data_points']} data points for BTC")
            print(f"   Sources used: {', '.join(result['metadata']['sources_used'])}")
        else:
            print("❌ No data collected for BTC")
        
        await collector.cleanup()
        
    except Exception as e:
        print(f"❌ Collector Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed! Ready for full data collection.")
    return True


async def main():
    """Main test execution"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║            Data Collection Setup Test                          ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    success = await test_setup()
    
    if success:
        print("\n\n🚀 Ready to run full collection with:")
        print("   python scripts/collect_all_ultimate_data.py")
    else:
        print("\n\n❌ Please fix the issues above before running full collection")


if __name__ == "__main__":
    asyncio.run(main())

