#!/usr/bin/env python3
"""
Discover All Assets Listed on Aster DEX
Tests rate limits and data availability without synthetic data.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import pandas as pd
import aiohttp
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_training.aster_dex_data_collector import AsterDEXDataCollector
from mcp_trader.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsterAssetDiscovery:
    """
    Discover and validate all assets listed on Aster DEX.
    Tests data availability and rate limits for training.
    """

    def __init__(self):
        self.settings = get_settings()
        self.collector = None
        self.session = None

        # Rate limiting
        self.request_delay = 0.1  # 100ms between requests
        self.max_requests_per_minute = 300  # Conservative limit

        # Data quality thresholds
        self.min_data_points = 100  # Minimum historical data points
        self.max_missing_pct = 0.1  # Maximum 10% missing data
        self.min_timeframe_days = 7  # Minimum 7 days of data

    async def initialize(self):
        """Initialize connections."""
        self.collector = AsterDEXDataCollector()
        await self.collector.initialize()

        self.session = aiohttp.ClientSession()
        logger.info("✅ Initialized Aster asset discovery")

    async def discover_all_assets(self) -> Dict[str, Dict]:
        """
        Discover all assets listed on Aster DEX.
        Returns comprehensive asset information.
        """
        logger.info("🔍 Discovering all Aster DEX assets...")

        # Test connectivity first
        connectivity = await self.test_connectivity()
        if not connectivity:
            logger.error("❌ Cannot connect to Aster DEX API")
            return {}

        # Get exchange info
        exchange_info = await self.get_exchange_info()
        if not exchange_info:
            logger.error("❌ Cannot get exchange information")
            return {}

        # Parse assets
        spot_assets = self.parse_spot_assets(exchange_info)
        perpetual_assets = self.parse_perpetual_assets(exchange_info)

        all_assets = {**spot_assets, **perpetual_assets}

        logger.info(f"📊 Discovered {len(all_assets)} total assets:")
        logger.info(f"   Spot: {len(spot_assets)}")
        logger.info(f"   Perpetual: {len(perpetual_assets)}")

        return all_assets

    async def test_connectivity(self) -> bool:
        """Test basic connectivity to Aster DEX."""
        collector = None
        try:
            collector = AsterDEXDataCollector()
            await collector.initialize()
            connected = await collector._test_connectivity()
            return connected
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            return False
        finally:
            if collector:
                await collector.close()

    async def get_exchange_info(self) -> Optional[Dict]:
        """Get comprehensive exchange information."""
        try:
            # Try multiple endpoints for exchange info
            # Start with the endpoint that works for connectivity test
            endpoints = [
                "https://fapi.asterdex.com/fapi/v1/exchangeInfo",  # This one works for connectivity
                "https://api.asterdex.com/api/v3/exchangeInfo",
                "https://api.asterdex.com/exchangeInfo",
                # Add other potential endpoints
            ]

            for endpoint in endpoints:
                try:
                    async with self.session.get(endpoint, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"✅ Got exchange info from {endpoint}")
                            return data
                        else:
                            logger.debug(f"Endpoint {endpoint} returned status {response.status}")
                except Exception as e:
                    logger.debug(f"Endpoint {endpoint} failed: {e}")
                    continue

            logger.error("❌ No exchange info endpoints available")
            return None

        except Exception as e:
            logger.error(f"Exchange info request failed: {e}")
            return None

    def parse_spot_assets(self, exchange_info: Dict) -> Dict[str, Dict]:
        """Parse spot trading assets."""
        spot_assets = {}

        if 'symbols' in exchange_info:
            for symbol in exchange_info['symbols']:
                if symbol.get('status') == 'TRADING' and symbol.get('quoteAsset') == 'USDT':
                    asset_info = {
                        'symbol': symbol['symbol'],
                        'base_asset': symbol['baseAsset'],
                        'quote_asset': symbol['quoteAsset'],
                        'type': 'spot',
                        'status': symbol['status'],
                        'filters': symbol.get('filters', []),
                        'permissions': symbol.get('permissions', []),
                    }
                    spot_assets[symbol['symbol']] = asset_info

        logger.info(f"📈 Found {len(spot_assets)} spot trading pairs")
        return spot_assets

    def parse_perpetual_assets(self, exchange_info: Dict) -> Dict[str, Dict]:
        """Parse perpetual futures assets."""
        perpetual_assets = {}

        # Look for perpetual symbols (might be in different section)
        if 'perpetualSymbols' in exchange_info:
            for symbol in exchange_info['perpetualSymbols']:
                if symbol.get('status') == 'TRADING':
                    asset_info = {
                        'symbol': symbol['symbol'],
                        'base_asset': symbol['baseAsset'],
                        'quote_asset': symbol.get('quoteAsset', 'USDT'),
                        'type': 'perpetual',
                        'status': symbol['status'],
                        'leverage': symbol.get('leverage', 1),
                        'funding_rate': symbol.get('fundingRate', 0),
                    }
                    perpetual_assets[symbol['symbol']] = asset_info

        logger.info(f"🔄 Found {len(perpetual_assets)} perpetual contracts")
        return perpetual_assets

    async def test_asset_data_availability(self, assets: Dict[str, Dict], sample_size: int = 20) -> Dict[str, Dict]:
        """
        Test data availability for a sample of assets.
        Returns only assets with sufficient data.
        """
        logger.info(f"🧪 Testing data availability for {min(sample_size, len(assets))} assets...")

        tested_assets = {}
        request_count = 0
        start_time = time.time()

        # Test a sample of assets
        test_symbols = list(assets.keys())[:sample_size]

        for symbol in test_symbols:
            # Rate limiting
            if request_count >= self.max_requests_per_minute:
                elapsed = time.time() - start_time
                if elapsed < 60:
                    await asyncio.sleep(60 - elapsed)
                request_count = 0
                start_time = time.time()

            try:
                # Test recent trades
                trades = await self.collector.collect_recent_trades(symbol, limit=10)
                has_trades = len(trades) > 0 if trades else False

                # Test orderbook
                orderbook = await self.collector.collect_orderbook_data(symbol)
                has_orderbook = orderbook is not None and 'bids' in orderbook and 'asks' in orderbook

                # Test historical data (last 24 hours)
                start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                end_date = datetime.now().strftime("%Y-%m-%d")

                hist_data = await self.collector.collect_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1h",
                    limit=100
                )

                data_points = len(hist_data) if not hist_data.empty else 0
                has_historical = data_points >= 10  # At least 10 hours of data

                # Check data quality
                data_quality = self.assess_data_quality(hist_data) if not hist_data.empty else 0

                asset_result = {
                    **assets[symbol],
                    'has_trades': has_trades,
                    'has_orderbook': has_orderbook,
                    'has_historical': has_historical,
                    'data_points_24h': data_points,
                    'data_quality_score': data_quality,
                    'is_trainable': has_trades and has_orderbook and has_historical and data_quality > 0.7
                }

                tested_assets[symbol] = asset_result

                status = "✅" if asset_result['is_trainable'] else "⚠️"
                logger.info(f"{status} {symbol}: Trades={has_trades}, Orderbook={has_orderbook}, "
                          f"Data={data_points}pts, Quality={data_quality:.2f}")

                request_count += 1
                await asyncio.sleep(self.request_delay)

            except Exception as e:
                logger.warning(f"❌ Failed to test {symbol}: {e}")
                tested_assets[symbol] = {
                    **assets[symbol],
                    'has_trades': False,
                    'has_orderbook': False,
                    'has_historical': False,
                    'data_points_24h': 0,
                    'data_quality_score': 0,
                    'is_trainable': False,
                    'error': str(e)
                }

        # Filter to trainable assets only
        trainable_assets = {k: v for k, v in tested_assets.items() if v['is_trainable']}
        logger.info(f"🎯 Found {len(trainable_assets)} trainable assets out of {len(tested_assets)} tested")

        return trainable_assets

    def assess_data_quality(self, df: pd.DataFrame) -> float:
        """
        Assess data quality score (0-1).
        Returns 0 if data is insufficient.
        """
        if df.empty or len(df) < self.min_data_points:
            return 0.0

        quality_score = 1.0

        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > self.max_missing_pct:
            quality_score *= (1 - missing_pct)

        # Check data completeness (all OHLCV columns present)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            quality_score *= 0.5

        # Check for reasonable price ranges (not all zeros)
        if (df['close'] == 0).all():
            quality_score *= 0.1

        # Check for extreme outliers (price changes > 50% in single period)
        returns = df['close'].pct_change().abs()
        extreme_changes = (returns > 0.5).sum()
        if extreme_changes > len(df) * 0.1:  # More than 10% extreme changes
            quality_score *= 0.8

        return max(0.0, quality_score)

    def get_trainable_assets_report(self, assets: Dict[str, Dict]) -> pd.DataFrame:
        """Generate comprehensive report of trainable assets."""
        report_data = []

        for symbol, info in assets.items():
            report_data.append({
                'symbol': symbol,
                'type': info.get('type', 'unknown'),
                'base_asset': info.get('base_asset', ''),
                'has_trades': info.get('has_trades', False),
                'has_orderbook': info.get('has_orderbook', False),
                'has_historical': info.get('has_historical', False),
                'data_points_24h': info.get('data_points_24h', 0),
                'data_quality_score': info.get('data_quality_score', 0),
                'is_trainable': info.get('is_trainable', False)
            })

        df = pd.DataFrame(report_data)
        return df.sort_values(['is_trainable', 'data_quality_score'], ascending=[False, False])

    async def save_asset_discovery_report(self, assets: Dict[str, Dict], output_file: str = "aster_assets_discovery.json"):
        """Save complete asset discovery report."""
        report = {
            'discovery_timestamp': datetime.now().isoformat(),
            'total_assets_discovered': len(assets),
            'trainable_assets': sum(1 for a in assets.values() if a.get('is_trainable', False)),
            'assets_by_type': {
                'spot': sum(1 for a in assets.values() if a.get('type') == 'spot'),
                'perpetual': sum(1 for a in assets.values() if a.get('type') == 'perpetual')
            },
            'assets': assets
        }

        output_path = Path("data") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"💾 Saved asset discovery report to {output_path}")
        return str(output_path)

    async def close(self):
        """Close connections."""
        if self.collector:
            await self.collector.close()
        if self.session:
            await self.session.close()


async def main():
    """Main execution."""
    print("""
================================================================================
           Aster DEX Asset Discovery & Validation
     Testing Rate Limits & Data Availability (No Synthetic)
================================================================================
    """)

    discovery = AsterAssetDiscovery()

    try:
        # Initialize
        await discovery.initialize()

        # Discover all assets
        all_assets = await discovery.discover_all_assets()

        if not all_assets:
            logger.error("❌ No assets discovered. Check API connectivity.")
            return

        # Test data availability for sample
        logger.info("\n🔬 Testing data availability and rate limits...")
        tested_assets = await discovery.test_asset_data_availability(all_assets, sample_size=50)

        # Generate reports
        report_df = discovery.get_trainable_assets_report(tested_assets)
        report_file = await discovery.save_asset_discovery_report(tested_assets)

        # Print summary
        print(f"\n{'='*70}")
        print("ASSET DISCOVERY SUMMARY")
        print(f"{'='*70}\n")

        print("📊 Overall Statistics:")
        print(f"   Total assets discovered: {len(all_assets)}")
        print(f"   Assets tested: {len(tested_assets)}")
        print(f"   Trainable assets: {len([a for a in tested_assets.values() if a.get('is_trainable')])}")
        print(f"   Spot assets: {len([a for a in all_assets.values() if a.get('type') == 'spot'])}")
        print(f"   Perpetual assets: {len([a for a in all_assets.values() if a.get('type') == 'perpetual'])}")

        print("\nTop 10 Trainable Assets:")
        top_assets = report_df.head(10)
        for _, row in top_assets.iterrows():
            print(f"   {row['symbol']:12} | Type: {row['type']:9} | Quality: {row['data_quality_score']:.2f}")

        print("\nReport saved:")
        print(f"   {report_file}")

        print("""
================================================================================
                    Discovery Complete!
     Ready for training with real Aster DEX data only
================================================================================

Next steps:
1. Review aster_assets_discovery.json for complete asset list
2. Run data collection for trainable assets only
3. Train models without synthetic data
4. Backtest with real market conditions
        """)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Discovery interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Discovery failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await discovery.close()


if __name__ == "__main__":
    asyncio.run(main())

