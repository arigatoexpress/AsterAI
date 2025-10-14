"""
Aster DEX Data Collector for Local ML Training
RTX 5070Ti optimized data collection and processing for Aster DEX trading pairs.
"""

import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AsterDEXDataCollector:
    """
    High-performance data collector for Aster DEX trading pairs.
    Optimized for RTX 5070Ti local ML training with parallel processing.
    """

    def __init__(self, api_key: str = None, api_secret: str = None, base_url: str = None):
        self.api_key = api_key or "demo_key"
        self.api_secret = api_secret or "demo_secret"
        self.base_url = base_url or "https://fapi.asterdex.com"

        # Data storage
        self.data_dir = Path.home() / "ai_trading_local" / "data"
        self.raw_data_dir = self.data_dir / "raw" / "aster_dex"
        self.processed_data_dir = self.data_dir / "processed" / "aster_dex"

        for dir_path in [self.raw_data_dir, self.processed_data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Trading pairs to collect
        self.priority_symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT",
            "ASTERUSDT", "PENGUUSDT"
        ]

        # Data collection settings
        self.max_concurrent_requests = 8  # For 16-core CPU
        self.request_timeout = 30
        self.retry_attempts = 3
        self.rate_limit_delay = 0.1  # 100ms between requests

        # Session for connection pooling
        self.session = None

        logger.info("Aster DEX Data Collector initialized")

    async def initialize(self):
        """Initialize HTTP session and validate connectivity."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.request_timeout),
            connector=aiohttp.TCPConnector(limit=self.max_concurrent_requests * 2)
        )

        # Test connectivity
        try:
            await self._test_connectivity()
            logger.info("✅ Aster DEX connectivity verified")
        except Exception as e:
            logger.warning(f"⚠️  Aster DEX connectivity test failed: {e}")
            logger.info("Continuing with demo mode")

    async def _test_connectivity(self):
        """Test Aster DEX API connectivity."""
        url = f"{self.base_url}/fapi/v1/exchangeInfo"

        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Connected to Aster DEX: {len(data.get('symbols', []))} symbols available")
            else:
                raise Exception(f"API returned status {response.status}")

    async def collect_historical_data(self, symbol: str, start_date: str, end_date: str,
                                    interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
        """
        Collect historical OHLCV data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            interval: Kline interval ("1m", "5m", "15m", "1h", "4h", "1d")
            limit: Maximum number of klines per request

        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Collecting historical data for {symbol} ({interval})")

            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

            all_klines = []
            current_ts = start_ts

            while current_ts < end_ts:
                # Calculate next batch end time (max 1000 klines per request)
                batch_end = min(current_ts + (limit * self._interval_to_ms(interval)), end_ts)

                klines = await self._get_klines_batch(symbol, interval, current_ts, batch_end, limit)

                if not klines:
                    break

                all_klines.extend(klines)

                # Move to next batch
                current_ts = batch_end

                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

            if not all_klines:
                logger.warning(f"No data collected for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = self._klines_to_dataframe(all_klines, interval)

            # Save raw data
            self._save_raw_data(symbol, interval, df)

            logger.info(f"✅ Collected {len(df)} klines for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error collecting historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def _get_klines_batch(self, symbol: str, interval: str, start_time: int,
                              end_time: int, limit: int) -> List[List]:
        """Get a batch of klines data."""
        url = f"{self.base_url}/fapi/v1/klines"

        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': min(limit, 1000)
        }

        for attempt in range(self.retry_attempts):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"HTTP {response.status} for {symbol}")
                        return []

            except Exception as e:
                logger.warning(f"Request failed for {symbol}, attempt {attempt}: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)

        return []

    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        multipliers = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return multipliers.get(interval, 60 * 60 * 1000)

    def _klines_to_dataframe(self, klines: List[List], interval: str) -> pd.DataFrame:
        """Convert klines data to DataFrame."""
        if not klines:
            return pd.DataFrame()

        # Kline format: [open_time, open, high, low, close, volume, ...]
        df_data = []
        for kline in klines:
            if len(kline) >= 6:
                df_data.append({
                    'timestamp': pd.to_datetime(kline[0], unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })

        df = pd.DataFrame(df_data)

        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

        return df

    def _save_raw_data(self, symbol: str, interval: str, df: pd.DataFrame):
        """Save raw data to disk."""
        if df.empty:
            return

        filename = f"{symbol}_{interval}_{df.index.min().strftime('%Y%m%d')}_{df.index.max().strftime('%Y%m%d')}.parquet"
        filepath = self.raw_data_dir / filename

        df.to_parquet(filepath)
        logger.debug(f"Saved raw data to {filepath}")

    async def collect_orderbook_data(self, symbol: str, depth: int = 100) -> Dict[str, Any]:
        """Collect current orderbook data."""
        url = f"{self.base_url}/fapi/v1/depth"

        params = {
            'symbol': symbol,
            'limit': depth
        }

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Orderbook request failed for {symbol}: HTTP {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error collecting orderbook for {symbol}: {e}")
            return {}

    async def collect_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Collect recent trades data."""
        url = f"{self.base_url}/fapi/v1/trades"

        params = {
            'symbol': symbol,
            'limit': limit
        }

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Trades request failed for {symbol}: HTTP {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error collecting trades for {symbol}: {e}")
            return []

    async def collect_all_symbols_data(self, days: int = 365, intervals: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect comprehensive data for all priority symbols.

        Args:
            days: Number of days of historical data to collect
            intervals: List of intervals to collect ["1h", "4h", "1d"]

        Returns:
            Nested dict: {symbol: {interval: dataframe}}
        """
        if intervals is None:
            intervals = ["1h", "4h", "1d"]

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        all_data = {}

        # Use semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def collect_symbol_data(symbol: str):
            async with semaphore:
                symbol_data = {}

                for interval in intervals:
                    df = await self.collect_historical_data(
                        symbol,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        interval
                    )

                    if not df.empty:
                        symbol_data[interval] = df

                return symbol, symbol_data

        # Collect data for all symbols in parallel
        tasks = [collect_symbol_data(symbol) for symbol in self.priority_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error collecting data: {result}")
                continue

            symbol, symbol_data = result
            if symbol_data:
                all_data[symbol] = symbol_data
                logger.info(f"✅ Collected data for {symbol}: {list(symbol_data.keys())}")

        logger.info(f"✅ Collected data for {len(all_data)} symbols")
        return all_data

    async def collect_real_time_data(self, symbol: str, duration_minutes: int = 60) -> pd.DataFrame:
        """
        Collect real-time price data for live training.

        Args:
            symbol: Trading pair symbol
            duration_minutes: How long to collect data

        Returns:
            DataFrame with real-time price data
        """
        logger.info(f"Starting real-time data collection for {symbol} ({duration_minutes} minutes)")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        price_data = []

        while time.time() < end_time:
            try:
                # Get current ticker
                ticker = await self._get_ticker(symbol)

                if ticker:
                    price_data.append({
                        'timestamp': datetime.now(),
                        'price': ticker.get('lastPrice', 0),
                        'volume': ticker.get('volume', 0),
                        'bid': ticker.get('bidPrice', 0),
                        'ask': ticker.get('askPrice', 0)
                    })

                # Wait 1 second
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in real-time collection: {e}")
                await asyncio.sleep(5)

        if price_data:
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)

            # Save real-time data
            filename = f"{symbol}_realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            filepath = self.raw_data_dir / filename
            df.to_parquet(filepath)

            logger.info(f"✅ Collected {len(df)} real-time data points for {symbol}")
            return df

        return pd.DataFrame()

    async def _get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current ticker data."""
        url = f"{self.base_url}/fapi/v1/ticker/24hr"

        params = {'symbol': symbol}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data[0] if isinstance(data, list) else data
                return None

        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None

    def process_and_save_data(self, all_data: Dict[str, Dict[str, pd.DataFrame]],
                            output_dir: str = None) -> Dict[str, str]:
        """
        Process and save collected data in optimized format.

        Args:
            all_data: Nested dict from collect_all_symbols_data
            output_dir: Custom output directory

        Returns:
            Dict mapping symbols to processed file paths
        """
        if output_dir is None:
            output_dir = self.processed_data_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_files = {}

        for symbol, intervals_data in all_data.items():
            try:
                # Combine all intervals for this symbol
                combined_df = pd.DataFrame()

                for interval, df in intervals_data.items():
                    # Add interval suffix to column names
                    interval_df = df.copy()
                    interval_df.columns = [f"{col}_{interval}" for col in interval_df.columns]

                    if combined_df.empty:
                        combined_df = interval_df
                    else:
                        # Merge on timestamp (assuming different intervals)
                        combined_df = combined_df.join(interval_df, how='outer')

                # Sort by timestamp
                if not combined_df.empty:
                    combined_df.index.name = 'timestamp'
                    combined_df = combined_df.sort_index()

                    # Save processed data
                    filename = f"{symbol}_processed_{datetime.now().strftime('%Y%m%d')}.parquet"
                    filepath = output_dir / filename
                    combined_df.to_parquet(filepath)

                    processed_files[symbol] = str(filepath)
                    logger.info(f"✅ Processed and saved {symbol} data: {len(combined_df)} rows")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        logger.info(f"✅ Processed data saved for {len(processed_files)} symbols")
        return processed_files

    def create_training_datasets(self, processed_files: Dict[str, str],
                               lookback_windows: List[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Create training datasets with different lookback windows.

        Args:
            processed_files: Dict from process_and_save_data
            lookback_windows: List of lookback periods in hours

        Returns:
            Dict of training datasets for each symbol and lookback
        """
        if lookback_windows is None:
            lookback_windows = [24, 72, 168, 336]  # 1d, 3d, 1w, 2w

        training_datasets = {}

        for symbol, filepath in processed_files.items():
            try:
                # Load processed data
                df = pd.read_parquet(filepath)

                if df.empty:
                    continue

                # Create datasets for each lookback window
                for lookback in lookback_windows:
                    # Filter to recent data
                    cutoff_date = df.index.max() - pd.Timedelta(hours=lookback)
                    recent_df = df[df.index >= cutoff_date]

                    if len(recent_df) >= lookback:
                        # Create features and targets
                        features = self._create_training_features(recent_df, lookback)
                        targets = self._create_training_targets(recent_df, lookback)

                        if not features.empty and not targets.empty:
                            training_datasets[f"{symbol}_{lookback}h"] = {
                                'features': features,
                                'targets': targets,
                                'metadata': {
                                    'symbol': symbol,
                                    'lookback_hours': lookback,
                                    'samples': len(features),
                                    'date_range': f"{recent_df.index.min()} to {recent_df.index.max()}"
                                }
                            }

            except Exception as e:
                logger.error(f"Error creating training dataset for {symbol}: {e}")

        logger.info(f"✅ Created {len(training_datasets)} training datasets")
        return training_datasets

    def _create_training_features(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Create feature matrix for training."""
        features = []

        # Rolling window features
        for i in range(lookback, len(df)):
            window_data = df.iloc[i-lookback:i]

            # Basic price features
            close_prices = window_data['close_1h'].dropna() if 'close_1h' in window_data.columns else window_data['close']

            if len(close_prices) >= 10:  # Minimum data points
                feature_vector = [
                    close_prices.mean(),  # Mean price
                    close_prices.std(),   # Volatility
                    close_prices.min(),   # Min price
                    close_prices.max(),   # Max price
                    (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0],  # Return
                ]

                # Volume features
                if 'volume_1h' in window_data.columns:
                    volume_data = window_data['volume_1h'].dropna()
                    if len(volume_data) > 0:
                        feature_vector.extend([
                            volume_data.mean(),
                            volume_data.sum()
                        ])

                features.append(feature_vector)

        return pd.DataFrame(features)

    def _create_training_targets(self, df: pd.DataFrame, lookback: int) -> pd.Series:
        """Create target values for training."""
        targets = []

        # Target: next hour return
        for i in range(lookback, len(df)):
            current_price = df.iloc[i]['close_1h'] if 'close_1h' in df.columns else df.iloc[i]['close']

            # Try to get next price
            if i + 1 < len(df):
                next_price = df.iloc[i+1]['close_1h'] if 'close_1h' in df.columns else df.iloc[i+1]['close']
                target_return = (next_price - current_price) / current_price
            else:
                target_return = 0.0  # No target for last point

            targets.append(target_return)

        return pd.Series(targets)

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            logger.info("✅ Aster DEX data collector closed")


# Example usage and testing
async def main():
    """Test Aster DEX data collection."""
    logging.basicConfig(level=logging.INFO)

    collector = AsterDEXDataCollector()
    await collector.initialize()

    try:
        # Test data collection for one symbol
        symbol = "BTCUSDT"

        # Collect historical data
        print(f"📊 Collecting historical data for {symbol}...")
        df = await collector.collect_historical_data(
            symbol,
            "2024-01-01",
            "2024-01-31",
            "1h",
            1000
        )

        if not df.empty:
            print(f"✅ Collected {len(df)} data points")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

        # Test orderbook collection
        print(f"📈 Collecting orderbook for {symbol}...")
        orderbook = await collector.collect_orderbook_data(symbol, 20)

        if orderbook:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            print(f"✅ Orderbook: {len(bids)} bids, {len(asks)} asks")

        # Test recent trades
        print(f"📋 Collecting recent trades for {symbol}...")
        trades = await collector.collect_recent_trades(symbol, 10)

        if trades:
            print(f"✅ Recent trades: {len(trades)} trades")
            if trades:
                latest_trade = trades[0]
                print(f"   Latest: ${latest_trade['price']} qty {latest_trade['qty']}")

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
