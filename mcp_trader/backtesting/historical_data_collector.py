import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import aiohttp
import json
import os
from pathlib import Path

from ..execution.aster_client import AsterClient
from ..config import get_settings

logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """Collects historical market data from Aster DEX for backtesting."""

    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings = get_settings()

    async def get_top_traded_pairs(self, limit: int = 20) -> List[str]:
        """Get the top traded pairs on Aster DEX."""
        try:
            async with AsterClient(self.settings.aster_api_key, self.settings.aster_api_secret) as client:
                # Get 24hr ticker statistics for all pairs
                tickers = await client.get_24hr_ticker_statistics()

                if not tickers:
                    logger.warning("No ticker data received")
                    return []

                # Sort by quote volume (trading volume)
                sorted_tickers = sorted(
                    tickers,
                    key=lambda x: float(x.get('quoteVolume', 0)),
                    reverse=True
                )

                # Extract symbols and ensure they end with USDT
                symbols = []
                for ticker in sorted_tickers[:limit]:
                    symbol = ticker.get('symbol', '')
                    if symbol and symbol.endswith('USDT'):
                        symbols.append(symbol)

                logger.info(f"Retrieved top {len(symbols)} traded pairs: {symbols}")
                return symbols

        except Exception as e:
            logger.error(f"Error getting top traded pairs: {e}")
            return []

    async def fetch_historical_klines(self,
                                    symbol: str,
                                    interval: str = '1h',
                                    start_date: datetime = None,
                                    end_date: datetime = None,
                                    limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical klines (candlestick data) from Aster DEX.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Start date for data collection
            end_date: End date for data collection
            limit: Maximum number of klines per request

        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = datetime(2024, 1, 1)  # Default to 2024 onwards
        if end_date is None:
            end_date = datetime.now()

        try:
            async with AsterClient(self.settings.aster_api_key, self.settings.aster_api_secret) as client:
                all_klines = []
                current_start = start_date

                while current_start < end_date:
                    # Calculate end time for this batch
                    batch_end = min(current_start + timedelta(days=30), end_date)

                    logger.info(f"Fetching {symbol} klines from {current_start} to {batch_end}")

                    try:
                        # Get klines for this time range
                        klines = await client.get_klines(
                            symbol=symbol,
                            interval=interval,
                            start_time=int(current_start.timestamp() * 1000),
                            end_time=int(batch_end.timestamp() * 1000),
                            limit=limit
                        )

                        if klines:
                            all_klines.extend(klines)
                            logger.info(f"Retrieved {len(klines)} klines for {symbol}")
                        else:
                            logger.warning(f"No klines received for {symbol} in range {current_start} - {batch_end}")

                        # Move to next batch
                        current_start = batch_end

                        # Rate limiting
                        await asyncio.sleep(0.1)

                    except Exception as e:
                        logger.error(f"Error fetching klines for {symbol}: {e}")
                        break

                # Convert to DataFrame
                if all_klines:
                    df = self._process_klines_to_dataframe(all_klines)
                    logger.info(f"Successfully collected {len(df)} total klines for {symbol}")
                    return df
                else:
                    logger.warning(f"No historical data collected for {symbol}")
                    return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in fetch_historical_klines for {symbol}: {e}")
            return pd.DataFrame()

    def _process_klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """Convert raw kline data to pandas DataFrame."""
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume']

        df = pd.DataFrame(klines, columns=columns)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert price and volume columns to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                          'taker_buy_volume', 'taker_buy_quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Sort by timestamp
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        return df

    async def collect_recent_market_data(self,
                                       symbols: List[str] = None,
                                       intervals: List[str] = None,
                                       days_back: int = 30) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect recent market data for adaptive learning.

        Args:
            symbols: List of trading pairs
            intervals: Time intervals to collect
            days_back: Number of days of recent data to collect

        Returns:
            Nested dict: {symbol: {interval: DataFrame}}
        """
        start_date = datetime.now() - timedelta(days=days_back)

        return await self.collect_all_historical_data(symbols, intervals, start_date)

    async def collect_all_historical_data(self,
                                        symbols: List[str] = None,
                                        intervals: List[str] = None,
                                        start_date: datetime = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect historical data for multiple symbols and intervals.

        Args:
            symbols: List of trading pairs, if None uses top 20
            intervals: List of intervals, if None uses ['1h', '4h', '1d']
            start_date: Start date, if None uses 2024-01-01

        Returns:
            Nested dict: {symbol: {interval: DataFrame}}
        """
        if symbols is None:
            symbols = await self.get_top_traded_pairs(20)

        if intervals is None:
            intervals = ['1h', '4h', '1d']

        if start_date is None:
            start_date = datetime(2024, 1, 1)

        historical_data = {}

        for symbol in symbols:
            historical_data[symbol] = {}
            logger.info(f"Collecting historical data for {symbol}")

            for interval in intervals:
                try:
                    df = await self.fetch_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_date=start_date
                    )

                    if not df.empty:
                        historical_data[symbol][interval] = df
                        logger.info(f"Collected {len(df)} {interval} klines for {symbol}")

                        # Save to disk
                        self.save_historical_data(symbol, interval, df)
                    else:
                        logger.warning(f"No data collected for {symbol} {interval}")

                except Exception as e:
                    logger.error(f"Error collecting {symbol} {interval}: {e}")

                # Rate limiting between requests
                await asyncio.sleep(0.5)

        return historical_data

    def save_historical_data(self, symbol: str, interval: str, df: pd.DataFrame):
        """Save historical data to disk."""
        try:
            symbol_dir = self.data_dir / symbol
            symbol_dir.mkdir(exist_ok=True)

            filename = f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = symbol_dir / filename

            df.to_csv(filepath)
            logger.info(f"Saved {len(df)} rows to {filepath}")

        except Exception as e:
            logger.error(f"Error saving data for {symbol} {interval}: {e}")

    def load_historical_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load historical data from disk."""
        try:
            symbol_dir = self.data_dir / symbol
            if not symbol_dir.exists():
                return None

            # Find the most recent file
            files = list(symbol_dir.glob(f"{symbol}_{interval}_*.csv"))
            if not files:
                return None

            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} rows from {latest_file}")
            return df

        except Exception as e:
            logger.error(f"Error loading data for {symbol} {interval}: {e}")
            return None

    async def update_historical_data(self, symbols: List[str] = None,
                                   intervals: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Update existing historical data with latest market data.
        This appends real-time data to existing historical datasets.
        """
        if symbols is None:
            symbols = await self.get_top_traded_pairs(20)

        if intervals is None:
            intervals = ['1h', '4h', '1d']

        updated_data = {}

        for symbol in symbols:
            updated_data[symbol] = {}

            for interval in intervals:
                try:
                    # Load existing data
                    existing_df = self.load_historical_data(symbol, interval)

                    if existing_df is not None and not existing_df.empty:
                        # Get the latest timestamp from existing data
                        last_timestamp = existing_df.index.max()

                        # Fetch new data from that timestamp onwards
                        new_df = await self.fetch_historical_klines(
                            symbol=symbol,
                            interval=interval,
                            start_date=last_timestamp.to_pydatetime(),
                            end_date=datetime.now()
                        )

                        if not new_df.empty:
                            # Remove any overlapping data
                            new_df = new_df[new_df.index > last_timestamp]

                            if not new_df.empty:
                                # Combine old and new data
                                combined_df = pd.concat([existing_df, new_df])
                                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                                combined_df.sort_index(inplace=True)

                                updated_data[symbol][interval] = combined_df
                                self.save_historical_data(symbol, interval, combined_df)

                                logger.info(f"Updated {symbol} {interval}: added {len(new_df)} new rows")
                            else:
                                updated_data[symbol][interval] = existing_df
                                logger.info(f"No new data for {symbol} {interval}")
                        else:
                            updated_data[symbol][interval] = existing_df
                            logger.info(f"No new data available for {symbol} {interval}")
                    else:
                        # No existing data, collect fresh
                        df = await self.fetch_historical_klines(symbol, interval)
                        if not df.empty:
                            updated_data[symbol][interval] = df
                            self.save_historical_data(symbol, interval, df)
                            logger.info(f"Collected fresh data for {symbol} {interval}: {len(df)} rows")

                except Exception as e:
                    logger.error(f"Error updating {symbol} {interval}: {e}")

        return updated_data


async def main():
    """Main function for testing the historical data collector."""
    logging.basicConfig(level=logging.INFO)

    collector = HistoricalDataCollector()

    # Test getting top pairs
    top_pairs = await collector.get_top_traded_pairs(5)
    print(f"Top 5 pairs: {top_pairs}")

    if top_pairs:
        # Test collecting historical data for one pair
        symbol = top_pairs[0]
        print(f"Collecting historical data for {symbol}...")

        df = await collector.fetch_historical_klines(symbol, '1h', datetime(2024, 6, 1))
        if not df.empty:
            print(f"Collected {len(df)} klines for {symbol}")
            print(df.head())
        else:
            print(f"No data collected for {symbol}")


if __name__ == "__main__":
    asyncio.run(main())
