#!/usr/bin/env python3
"""
Streamlined Synchronous Data Collection for Aster DEX
Direct collection using known working endpoints - no async/asyncio issues.
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyncAsterDataCollector:
    """Synchronous data collector for Aster DEX."""

    def __init__(self, output_dir=None):
        self.base_url = "https://fapi.asterdex.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AsterAI-Trading-Bot/1.0',
            'Accept': 'application/json'
        })

        # Data directories
        self.data_dir = Path(output_dir) if output_dir else Path("data/historical/real_aster_only")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Known working perpetual symbols (from API testing)
        self.priority_symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT",
            "LINKUSDT", "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "COMPUSDT",
            "AVAXUSDT", "MATICUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"
        ]

        # Collection settings
        self.max_requests_per_minute = 60  # Conservative rate limiting
        self.request_interval = 60 / self.max_requests_per_minute
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            wait_time = self.request_interval - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def get_exchange_info(self) -> Optional[Dict]:
        """Get exchange information synchronously."""
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        self._rate_limit_wait()

        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                symbols = data.get('symbols', [])
                logger.info(f"✅ Got exchange info: {len(symbols)} symbols available")
                return data
            else:
                logger.error(f"❌ Exchange info failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"❌ Exchange info error: {e}")
            return None

    def get_historical_klines(self, symbol: str, interval: str = "1h",
                            start_date: str = None, end_date: str = None,
                            limit: int = 1000) -> pd.DataFrame:
        """Get historical klines synchronously."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Convert to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        all_klines = []
        current_ts = start_ts

        logger.info(f"📊 Collecting {symbol} {interval} data from {start_date} to {end_date}")

        while current_ts < end_ts:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_ts,
                'limit': min(limit, 1000)  # API limit
            }

            self._rate_limit_wait()

            try:
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    klines = response.json()
                    if not klines:
                        break

                    all_klines.extend(klines)

                    # Update current_ts to last kline time + 1ms
                    last_time = klines[-1][0]
                    current_ts = last_time + 1

                    logger.info(f"   📈 Got {len(klines)} klines for {symbol} (total: {len(all_klines)})")

                    # Break if we got less than requested (end of data)
                    if len(klines) < limit:
                        break

                else:
                    logger.warning(f"   ⚠️  API error {response.status_code} for {symbol}")
                    break

            except Exception as e:
                logger.error(f"   ❌ Error collecting {symbol}: {e}")
                break

        # Convert to DataFrame
        if all_klines:
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.drop(['close_time', 'trades', 'taker_buy_quote_volume', 'ignore'], axis=1)
            df = df.set_index('timestamp').sort_index()

            logger.info(f"✅ Collected {len(df)} rows for {symbol}")
            return df
        else:
            logger.warning(f"❌ No data collected for {symbol}")
            return pd.DataFrame()

    def get_recent_trades(self, symbol: str, limit: int = 100) -> Optional[List]:
        """Get recent trades synchronously."""
        url = f"{self.base_url}/fapi/v1/trades"
        params = {'symbol': symbol, 'limit': min(limit, 1000)}

        self._rate_limit_wait()

        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                trades = response.json()
                logger.info(f"✅ Got {len(trades)} recent trades for {symbol}")
                return trades
            else:
                logger.warning(f"❌ Recent trades failed for {symbol}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"❌ Recent trades error for {symbol}: {e}")
            return None

    def get_orderbook(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Get orderbook synchronously."""
        url = f"{self.base_url}/fapi/v1/depth"
        params = {'symbol': symbol, 'limit': min(limit, 1000)}

        self._rate_limit_wait()

        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                orderbook = response.json()
                bids = len(orderbook.get('bids', []))
                asks = len(orderbook.get('asks', []))
                logger.info(f"✅ Got orderbook for {symbol}: {bids} bids, {asks} asks")
                return orderbook
            else:
                logger.warning(f"❌ Orderbook failed for {symbol}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"❌ Orderbook error for {symbol}: {e}")
            return None

    def collect_asset_data(self, symbol: str, timeframes: List[str] = None) -> Dict[str, Any]:
        """Collect all data types for a single asset."""
        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]

        logger.info(f"🔄 Collecting data for {symbol}...")

        asset_data = {
            'symbol': symbol,
            'success': False,
            'timeframes': {},
            'trades': None,
            'orderbook': None,
            'quality_score': 0.0,
            'data_points': 0,
            'error': None
        }

        try:
            # Collect historical data for each timeframe
            total_points = 0
            successful_timeframes = 0

            for tf in timeframes:
                df = self.get_historical_klines(symbol, tf)
                if not df.empty:
                    asset_data['timeframes'][tf] = df
                    total_points += len(df)
                    successful_timeframes += 1
                    logger.info(f"   ✅ {tf}: {len(df)} points")
                else:
                    logger.warning(f"   ❌ {tf}: No data")

            # Collect orderbook
            orderbook = self.get_orderbook(symbol)
            if orderbook:
                asset_data['orderbook'] = orderbook

            # Collect recent trades
            trades = self.get_recent_trades(symbol)
            if trades:
                asset_data['trades'] = trades

            # Calculate quality score
            if successful_timeframes > 0:
                completeness = successful_timeframes / len(timeframes)
                data_density = min(1.0, total_points / (successful_timeframes * 100))  # Expected 100 points per tf
                has_orderbook = 1.0 if orderbook else 0.0
                has_trades = 1.0 if trades else 0.0

                asset_data['quality_score'] = (
                    completeness * 0.4 +
                    data_density * 0.3 +
                    has_orderbook * 0.2 +
                    has_trades * 0.1
                )
                asset_data['data_points'] = total_points
                asset_data['success'] = True

                logger.info(f"   🎯 Quality score: {asset_data['quality_score']:.2f} ({total_points} points)")

            return asset_data

        except Exception as e:
            logger.error(f"❌ Failed to collect {symbol}: {e}")
            asset_data['error'] = str(e)
            return asset_data

    def save_asset_data(self, asset_data: Dict[str, Any]):
        """Save collected asset data to files."""
        symbol = asset_data['symbol']

        # Save each timeframe
        for tf, df in asset_data.get('timeframes', {}).items():
            filename = f"{symbol}_{tf}.parquet"
            filepath = self.data_dir / filename
            df.to_parquet(filepath)
            logger.info(f"💾 Saved {filepath}")

        # Save orderbook
        if asset_data.get('orderbook'):
            filename = f"{symbol}_orderbook.json"
            filepath = self.data_dir / filename
            with open(filepath, 'w') as f:
                json.dump(asset_data['orderbook'], f, indent=2)

        # Save trades
        if asset_data.get('trades'):
            filename = f"{symbol}_trades.json"
            filepath = self.data_dir / filename
            with open(filepath, 'w') as f:
                json.dump(asset_data['trades'], f, indent=2)

    def run_collection(self, symbols: List[str] = None, max_assets: int = 10):
        """Run data collection for multiple assets."""
        if symbols is None:
            symbols = self.priority_symbols[:max_assets]

        print("""
================================================================================
      Synchronous Aster DEX Data Collection
      Direct collection using working futures API endpoints
================================================================================
        """)

        logger.info(f"🎯 Starting collection for {len(symbols)} assets: {', '.join(symbols)}")

        # Test connectivity first
        logger.info("🔗 Testing connectivity...")
        exchange_info = self.get_exchange_info()
        if not exchange_info:
            logger.error("❌ Cannot connect to Aster DEX API")
            return {}

        results = {}
        successful = 0
        total_quality = 0.0

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n📊 [{i}/{len(symbols)}] Processing {symbol}...")

            asset_data = self.collect_asset_data(symbol)
            results[symbol] = asset_data

            if asset_data['success']:
                successful += 1
                total_quality += asset_data['quality_score']
                self.save_asset_data(asset_data)
                logger.info(f"✅ {symbol} completed - Quality: {asset_data['quality_score']:.2f}")
            else:
                logger.warning(f"❌ {symbol} failed - {asset_data.get('error', 'Unknown error')}")

        # Create summary report
        summary = {
            'collection_timestamp': datetime.now().isoformat(),
            'total_assets_attempted': len(symbols),
            'successful_collections': successful,
            'success_rate': successful / len(symbols) if symbols else 0,
            'average_quality': total_quality / successful if successful > 0 else 0,
            'total_data_points': sum(r.get('data_points', 0) for r in results.values()),
            'assets': results
        }

        # Save summary
        summary_file = self.data_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"💾 Summary saved to {summary_file}")

        # Print results
        print(f"\n{'='*80}")
        print("COLLECTION RESULTS SUMMARY")
        print(f"{'='*80}\n")

        print(f"📊 Assets Processed: {len(symbols)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {len(symbols) - successful}")
        print(f"📈 Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"⭐ Average Quality: {summary['average_quality']:.2f}")
        print(f"📈 Total Data Points: {summary['total_data_points']}")

        if successful > 0:
            best_assets = sorted(
                [(k, v) for k, v in results.items() if v.get('success')],
                key=lambda x: x[1]['quality_score'],
                reverse=True
            )[:5]

            print(f"\n🏆 Top 5 Assets by Quality:")
            for symbol, data in best_assets:
                print(f"   🥇 {symbol}: {data['quality_score']:.2f} ({data['data_points']} points)")

        print(f"""
================================================================================
                🎉 Collection Complete!
      {successful} assets collected successfully with real Aster DEX data
================================================================================

Data saved to: {self.data_dir}

Next steps:
1. Review collection_summary.json for detailed results
2. Proceed to feature engineering: python scripts/validate_confluence_features.py
3. Train models: python local_training/train_confluence_model.py
4. Backtest: python scripts/backtest_confluence_strategy.py
        """)

        return summary


def collect_aster_data_sync(symbols=None, output_dir=None):
    """
    Collect Aster DEX data synchronously.
    
    Args:
        symbols: List of symbols to collect (e.g., ['BTC', 'ETH'])
        output_dir: Output directory path
        
    Returns:
        Collection summary dict
    """
    collector = SyncAsterDataCollector(output_dir=output_dir)
    
    # If specific symbols provided, collect only those
    if symbols:
        summary = {'successful_collections': 0, 'failed_collections': 0, 'results': {}}
        for symbol in symbols:
            trading_symbol = f"{symbol}USDT"
            result = collector.collect_asset_data(trading_symbol)
            if result.get('status') == 'success':
                summary['successful_collections'] += 1
                summary['results'][trading_symbol] = result
            else:
                summary['failed_collections'] += 1
        return summary
    else:
        # Otherwise collect all available
        return collector.run_collection(max_assets=10)


def main():
    """Main execution."""
    collector = SyncAsterDataCollector()
    summary = collector.run_collection(max_assets=10)  # Collect 10 assets first

    if summary and summary.get('successful_collections', 0) > 0:
        logger.info("✅ Data collection completed successfully")
    else:
        logger.error("❌ Data collection failed")


if __name__ == "__main__":
    main()


