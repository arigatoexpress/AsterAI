#!/usr/bin/env python3
"""
Test Data Collection with Known Aster DEX Assets
Bypasses discovery step, tests collection directly.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_training.aster_dex_data_collector import AsterDEXDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known Aster DEX assets (based on connectivity test showing 200 symbols)
KNOWN_ASTER_ASSETS = [
    "BTCUSDT",    # Bitcoin
    "ETHUSDT",    # Ethereum
    "SOLUSDT",    # Solana
    "SUIUSDT",    # Sui
    "ASTERUSDT",  # Aster (native)
    "BNBUSDT",    # Binance Coin
    "ADAUSDT",    # Cardano
    "DOTUSDT",    # Polkadot
    "AVAXUSDT",   # Avalanche
    "MATICUSDT",  # Polygon
    "LINKUSDT",   # Chainlink
    "UNIUSDT",    # Uniswap
    "AAVEUSDT",   # Aave
    "SUSHIUSDT",  # SushiSwap
    "COMPUSDT",   # Compound
]


class KnownAssetsTester:
    """Test data collection with known Aster DEX assets."""

    def __init__(self):
        self.collector = None
        self.test_results = {}
        self.start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")  # 30 days
        self.end_date = datetime.now().strftime("%Y-%m-%d")

    async def initialize(self):
        """Initialize the collector."""
        self.collector = AsterDEXDataCollector()
        await self.collector.initialize()
        logger.info("✅ Known assets tester initialized")

    async def test_assets_availability(self, max_assets: int = 5):
        """Test which known assets are available for data collection."""
        logger.info(f"🔍 Testing availability of {min(max_assets, len(KNOWN_ASTER_ASSETS))} known assets...")

        available_assets = []
        test_assets = KNOWN_ASTER_ASSETS[:max_assets]

        for symbol in test_assets:
            logger.info(f"   Testing {symbol}...")

            try:
                # Test orderbook (fastest test)
                orderbook = await self.collector.collect_orderbook_data(symbol)
                has_orderbook = orderbook is not None and 'bids' in orderbook and len(orderbook.get('bids', [])) > 0

                # Test recent trades
                trades = await self.collector.collect_recent_trades(symbol, limit=5)
                has_trades = trades is not None and len(trades) > 0

                # Test historical data (1 day)
                hist_data = await self.collector.collect_historical_data(
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval="1h",
                    limit=100
                )
                has_historical = not hist_data.empty and len(hist_data) > 10

                asset_info = {
                    'symbol': symbol,
                    'has_orderbook': has_orderbook,
                    'has_trades': has_trades,
                    'has_historical': has_historical,
                    'historical_points': len(hist_data) if not hist_data.empty else 0,
                    'is_available': has_orderbook and has_trades and has_historical
                }

                self.test_results[symbol] = asset_info

                status = "✅" if asset_info['is_available'] else "❌"
                logger.info(f"     {status} {symbol}: OB={has_orderbook}, Trades={has_trades}, Hist={has_historical} ({asset_info['historical_points']} pts)")

                if asset_info['is_available']:
                    available_assets.append(symbol)

                # Rate limiting
                await asyncio.sleep(0.2)

            except Exception as e:
                logger.warning(f"     ⚠️  {symbol} failed: {e}")
                self.test_results[symbol] = {
                    'symbol': symbol,
                    'has_orderbook': False,
                    'has_trades': False,
                    'has_historical': False,
                    'historical_points': 0,
                    'is_available': False,
                    'error': str(e)
                }

        logger.info(f"\n📊 Asset Availability Summary:")
        logger.info(f"   Tested: {len(test_assets)} assets")
        logger.info(f"   Available: {len(available_assets)} assets")
        logger.info(f"   Success rate: {len(available_assets)/len(test_assets)*100:.1f}%")

        if available_assets:
            logger.info(f"   ✅ Available assets: {', '.join(available_assets[:5])}{'...' if len(available_assets) > 5 else ''}")

        return available_assets

    async def test_data_collection_quality(self, assets: list):
        """Test data collection quality for available assets."""
        logger.info(f"\n🔬 Testing data collection quality for {len(assets)} assets...")

        quality_results = {}

        for symbol in assets[:3]:  # Test first 3 available assets
            logger.info(f"   Quality testing {symbol}...")

            try:
                # Collect 7 days of data
                start_7d = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                end_7d = datetime.now().strftime("%Y-%m-%d")

                data_7d = await self.collector.collect_historical_data(
                    symbol=symbol,
                    start_date=start_7d,
                    end_date=end_7d,
                    interval="1h",
                    limit=200
                )

                if data_7d.empty:
                    logger.warning(f"     No data collected for {symbol}")
                    continue

                # Quality metrics
                total_points = len(data_7d)
                expected_points = 7 * 24  # 7 days * 24 hours
                completeness = total_points / expected_points

                # Check for missing values
                missing_pct = data_7d.isnull().sum().sum() / (len(data_7d) * len(data_7d.columns))

                # Check price reasonableness
                price_range = data_7d['close'].max() / data_7d['close'].min()
                price_volatility = data_7d['close'].pct_change().std()

                # Quality score (0-1)
                quality_score = min(1.0, (
                    completeness * 0.4 +
                    (1 - missing_pct) * 0.3 +
                    (1 if 0.5 < price_range < 5.0 else 0) * 0.2 +
                    (1 if price_volatility < 0.1 else 0) * 0.1
                ))

                quality_results[symbol] = {
                    'total_points': total_points,
                    'expected_points': expected_points,
                    'completeness': completeness,
                    'missing_pct': missing_pct,
                    'price_range': price_range,
                    'price_volatility': price_volatility,
                    'quality_score': quality_score,
                    'data_quality': 'EXCELLENT' if quality_score > 0.8 else 'GOOD' if quality_score > 0.6 else 'POOR'
                }

                logger.info(f"     📊 {symbol}: {total_points}/{expected_points} points ({completeness:.1%}), Quality: {quality_results[symbol]['data_quality']}")

                await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"     ❌ Quality test failed for {symbol}: {e}")

        return quality_results

    def create_mock_discovery_file(self, available_assets: list):
        """Create a mock discovery file for testing."""
        logger.info("📝 Creating mock discovery file for testing...")

        # Create mock discovery data
        mock_discovery = {
            'discovery_timestamp': datetime.now().isoformat(),
            'total_assets_discovered': len(available_assets),
            'trainable_assets': len(available_assets),
            'assets_by_type': {
                'spot': len(available_assets),
                'perpetual': 0
            },
            'assets': {}
        }

        # Add asset details
        for symbol in available_assets:
            mock_discovery['assets'][symbol] = {
                'symbol': symbol,
                'base_asset': symbol.replace('USDT', ''),
                'quote_asset': 'USDT',
                'type': 'spot',
                'status': 'TRADING',
                'is_trainable': True,
                'has_trades': True,
                'has_orderbook': True,
                'has_historical': True,
                'data_points_24h': 24,  # Mock
                'data_quality_score': 0.9,
                'filters': [],
                'permissions': ['SPOT']
            }

        # Save to data directory
        output_path = Path("data/aster_assets_discovery_mock.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_path, 'w') as f:
            json.dump(mock_discovery, f, indent=2, default=str)

        logger.info(f"✅ Mock discovery file saved to {output_path}")
        return str(output_path)

    async def run_full_test(self):
        """Run complete test suite."""
        print("""
================================================================================
      Known Aster DEX Assets Testing
      Direct Collection Test (Bypassing Discovery)
================================================================================
        """)

        try:
            await self.initialize()

            # Test asset availability
            available_assets = await self.test_assets_availability(max_assets=10)

            if not available_assets:
                logger.error("❌ No assets available for data collection")
                return {}

            # Test data quality
            quality_results = await self.test_data_collection_quality(available_assets)

            # Create mock discovery file
            mock_file = self.create_mock_discovery_file(available_assets)

            # Summary
            print(f"\n{'='*80}")
            print("TEST RESULTS SUMMARY")
            print(f"{'='*80}\n")

            print(f"📊 Assets Tested: {len(self.test_results)}")
            print(f"📈 Available Assets: {len(available_assets)}")
            print(f"📋 Available Symbols: {', '.join(available_assets)}")

            if quality_results:
                avg_quality = sum(r['quality_score'] for r in quality_results.values()) / len(quality_results)
                print(f"⭐ Average Quality Score: {avg_quality:.2f}")
                print(f"📈 Best Quality Asset: {max(quality_results.items(), key=lambda x: x[1]['quality_score'])[0]}")

            print(f"\n💾 Mock Discovery File: {mock_file}")

            print("""
================================================================================
                 🎉 Assets Testing Complete!
      Ready to proceed with data collection using available assets
================================================================================

Next steps:
1. Use mock discovery file for pipeline testing
2. Run data collection: python scripts/collect_real_aster_data.py
3. Train models: python local_training/train_confluence_model.py
4. Backtest: python scripts/backtest_confluence_strategy.py
            """)

            return {
                'available_assets': available_assets,
                'quality_results': quality_results,
                'mock_discovery_file': mock_file
            }

        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
        finally:
            if self.collector:
                await self.collector.close()


async def main():
    """Main execution."""
    tester = KnownAssetsTester()
    results = await tester.run_full_test()

    if results:
        logger.info("✅ Known assets testing completed successfully")
    else:
        logger.error("❌ Known assets testing failed")


if __name__ == "__main__":
    asyncio.run(main())



