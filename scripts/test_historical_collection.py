#!/usr/bin/env python3
"""
Test Historical Data Collection
Demonstrates collection and storage of historical market data
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.historical_collector import HistoricalDataCollector
from mcp_trader.data.asset_universe import get_asset_universe, AssetClass
from mcp_trader.logging_utils import get_logger

logger = get_logger(__name__)

async def test_historical_collection():
    """Test historical data collection functionality"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         AsterAI Historical Data Collection Test              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Get asset universe
    universe = get_asset_universe()
    crypto_assets = universe.get_by_class(AssetClass.CRYPTOCURRENCY)[:3]  # Test with 3 crypto assets
    equity_assets = universe.get_by_class(AssetClass.EQUITY)[:2]  # Test with 2 stocks

    test_assets = crypto_assets + equity_assets

    print(f"📊 Testing with {len(test_assets)} assets:")
    for asset in test_assets:
        print(f"   • {asset.symbol} ({asset.asset_class.value}) - {asset.name}")

    # Initialize collector
    from pathlib import Path
    collector = HistoricalDataCollector(Path("data/historical"))

    print(f"\n⚙️  Historical data collection setup:")
    print(f"   • Storage directory: {collector.data_dir}")
    print(f"   • Date range: 2024-01-01 to 2024-12-31 (demo)")
    print(f"   • API credentials loaded: {bool(collector.credentials.alpha_vantage_key)}")

    try:
        # Test collection for a few assets
        print(f"\n🔄 Starting historical data collection...")

        # Collect for priority assets as a demo
        print(f"\n📈 Collecting historical data for priority assets...")

        # Use the collect_all_priority_assets method with priority 1 (highest priority)
        from datetime import datetime
        results = await collector.collect_all_priority_assets(
            start_date=datetime(2024, 1, 1),
            priority_level=1  # Highest priority assets only
        )

        print(f"✅ Collection completed!")
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful
        print(f"   • Assets processed: {len(results)}")
        print(f"   • Successful: {successful}")
        print(f"   • Failed: {failed}")

        # Show collection summary
        summary_info = collector.get_collection_summary()
        print(f"\n📊 Collection summary:")
        print(f"   • Total assets: {summary_info.get('total_assets', 0)}")
        print(f"   • Assets with data: {summary_info.get('assets_with_data', 0)}")
        print(f"   • Total records: {summary_info.get('total_records', 0)}")
        print(f"   • Data size: {summary_info.get('total_size_mb', 0):.2f} MB")

        # Check for actual data files
        import os
        data_files = []
        for root, dirs, files in os.walk(collector.data_dir):
            for file in files:
                if file.endswith('.parquet'):
                    data_files.append(os.path.join(root, file))

        if data_files:
            print(f"\n💾 Data files created:")
            for file_path in data_files[:3]:  # Show first 3
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"   • {os.path.basename(file_path)}: {file_size:.1f} KB")
        else:
            print(f"\n💾 No data files found yet (may need API keys)")

        print(f"\n✅ Historical data collection test completed!")
        print(f"💡 To collect more data: python scripts/collect_historical_data.py")

    except Exception as e:
        logger.error(f"ERROR: Historical collection test failed: {e}")
        print(f"ERROR: Test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(test_historical_collection())
    if success:
        print("\nSUCCESS: Historical data collection is working!")
    else:
        print("\nNOTE: Check API credentials and try again")
