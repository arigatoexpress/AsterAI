#!/usr/bin/env python3
"""
Simplified data collection script
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.api_manager import APIKeyManager

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def print_banner():
    print("""
╔════════════════════════════════════════════════════════════════╗
║            Simplified Data Collection Test                     ║
╚════════════════════════════════════════════════════════════════╝
    """)


def collect_yahoo_data():
    """Collect data using yfinance"""
    print("\n📊 Testing Yahoo Finance data collection...")
    
    # Test symbols
    symbols = ['BTC-USD', 'ETH-USD', 'SPY', 'GLD', 'AAPL']
    
    output_dir = Path("data/historical/test_collection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol in symbols:
        try:
            print(f"\nCollecting {symbol}...")
            
            # Download data
            data = yf.download(
                symbol,
                start=datetime.now() - timedelta(days=30),
                end=datetime.now(),
                interval='1h',
                progress=False
            )
            
            if not data.empty:
                # Save to parquet
                output_file = output_dir / f"{symbol.replace('-', '_')}.parquet"
                data.to_parquet(output_file)
                print(f"  ✅ Saved {len(data)} records to {output_file}")
            else:
                print(f"  ❌ No data received for {symbol}")
                
        except Exception as e:
            print(f"  ❌ Error collecting {symbol}: {e}")
    
    print(f"\n✅ Data saved to: {output_dir}")


async def test_aster_connection():
    """Test Aster DEX connection"""
    print("\n🔌 Testing Aster DEX connection...")
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Test Aster API endpoint
            url = "https://fapi.asterdex.com/fapi/v1/time"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✅ Aster DEX connected! Server time: {data.get('serverTime', 'Unknown')}")
                else:
                    print(f"  ❌ Aster DEX returned status: {response.status}")
                    
    except Exception as e:
        print(f"  ❌ Aster DEX connection error: {e}")


async def main():
    """Main execution"""
    print_banner()
    
    # Load API keys
    print("\n🔑 Loading API keys...")
    api_manager = APIKeyManager()
    credentials = api_manager.load_credentials()
    
    validation = credentials.validate()
    configured_apis = sum(1 for v in validation.values() if v)
    print(f"  ✅ {configured_apis}/6 APIs configured")
    
    # Test Yahoo Finance
    collect_yahoo_data()
    
    # Test Aster connection
    await test_aster_connection()
    
    print("\n\n✅ Test collection complete!")
    print("\nTo run the full Ultimate Data Collection:")
    print("  python scripts/collect_all_ultimate_data.py")
    
    # Show what's available
    output_dir = Path("data/historical/test_collection")
    if output_dir.exists():
        files = list(output_dir.glob("*.parquet"))
        if files:
            print(f"\n📁 Test data files created: {len(files)}")
            for f in files[:5]:  # Show first 5
                print(f"  - {f.name}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Collection interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

