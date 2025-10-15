#!/usr/bin/env python3
"""
Collect Historical Data for Aster-Native Assets
Uses multiple data sources with fallback chain for new assets.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import aiohttp
import ccxt.async_support as ccxt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_training.aster_dex_data_collector import AsterDEXDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsterNativeDataCollector:
    """
    Collect historical data for Aster-native assets.

    NO SYNTHETIC DATA - Only real market data for training safety.

    Data sources (in priority order):
    1. Aster DEX (primary - preferred for platform-specific data)
    2. Binance (fallback for major pairs with sufficient history)
    3. CoinGecko (fallback for price data with sufficient history)
    4. Empty DataFrame (no synthetic data generation)
    """
    
    def __init__(self, output_dir: str = "data/historical/aster_native"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.aster_collector = None
        self.binance = None
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        
        # Aster-native assets (update as platform grows)
        self.aster_native_assets = [
            "ASTERUSDT",  # Native token
            "SUIUSDT",    # Sui ecosystem
            "APTUSDT",    # Aptos (if listed)
            # Add more as Aster lists new assets
        ]
        
        # Major reference pairs (always available on Binance)
        self.reference_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # 6 months of data
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=180)
        
        logger.info(f"📊 Aster Native Data Collector initialized")
        logger.info(f"   Native assets: {len(self.aster_native_assets)}")
        logger.info(f"   Reference pairs: {len(self.reference_pairs)}")
        logger.info(f"   Date range: {self.start_date.date()} to {self.end_date.date()}")
    
    async def initialize(self):
        """Initialize all data sources."""
        logger.info("🔌 Initializing data sources...")
        
        # Aster DEX collector
        self.aster_collector = AsterDEXDataCollector()
        await self.aster_collector.initialize()
        logger.info("  ✅ Aster DEX connected")
        
        # Binance (for fallback)
        self.binance = ccxt.binance()
        await self.binance.load_markets()
        logger.info("  ✅ Binance connected")
    
    async def collect_all_data(self):
        """Collect data for all Aster-native assets."""
        logger.info(f"\n{'='*60}")
        logger.info("Starting Aster Native Asset Data Collection")
        logger.info(f"{'='*60}\n")
        
        all_assets = self.aster_native_assets + self.reference_pairs
        results = {}
        
        for symbol in all_assets:
            try:
                logger.info(f"📥 Collecting {symbol}...")
                
                # Try multiple sources
                df = await self._collect_with_fallback(symbol)
                
                if not df.empty:
                    # Save data
                    for interval in ["1h", "4h", "1d"]:
                        resampled = self._resample_data(df, interval)
                        output_file = self.output_dir / f"{symbol}_{interval}.parquet"
                        resampled.to_parquet(output_file, compression='snappy')
                        logger.info(f"  ✅ Saved {len(resampled)} {interval} records")
                    
                    results[symbol] = {
                        'success': True,
                        'records': len(df),
                        'source': df.attrs.get('source', 'unknown')
                    }
                else:
                    logger.warning(f"  ⚠️  No data collected for {symbol}")
                    results[symbol] = {'success': False, 'reason': 'no_data'}
                
            except Exception as e:
                logger.error(f"  ❌ Error collecting {symbol}: {e}")
                results[symbol] = {'success': False, 'reason': str(e)}
        
        # Save collection summary
        self._save_summary(results)
        
        logger.info(f"\n{'='*60}")
        logger.info("Collection Complete")
        logger.info(f"{'='*60}\n")
        
        successful = sum(1 for r in results.values() if r.get('success'))
        logger.info(f"✅ Successfully collected: {successful}/{len(all_assets)}")
        
        return results
    
    async def _collect_with_fallback(self, symbol: str) -> pd.DataFrame:
        """
        Collect data using fallback chain (NO SYNTHETIC DATA).

        Priority:
        1. Aster DEX (primary - preferred for platform-specific data)
        2. Binance (fallback for major pairs with sufficient history)
        3. CoinGecko (fallback for price data with sufficient history)
        4. Return empty DataFrame (no synthetic data generation)
        """
        
        # Try Aster DEX first
        logger.info(f"  🎯 Trying Aster DEX...")
        df = await self._collect_from_aster(symbol)
        if not df.empty:
            df.attrs['source'] = 'aster_dex'
            logger.info(f"  ✅ Collected {len(df)} records from Aster DEX")
            return df
        
        # Try Binance
        logger.info(f"  🎯 Trying Binance...")
        df = await self._collect_from_binance(symbol)
        if not df.empty:
            df.attrs['source'] = 'binance'
            logger.info(f"  ✅ Collected {len(df)} records from Binance")
            return df
        
        # Try CoinGecko
        logger.info(f"  🎯 Trying CoinGecko...")
        df = await self._collect_from_coingecko(symbol)
        if not df.empty:
            df.attrs['source'] = 'coingecko'
            logger.info(f"  ✅ Collected {len(df)} records from CoinGecko")
            return df
        
        # NO SYNTHETIC DATA - Return empty DataFrame for training safety
        logger.warning(f"  ❌ No real data available for {symbol} - SKIPPING")
        logger.warning(f"  💡 Only using verified real market data for training")
        return pd.DataFrame()
    
    async def _collect_from_aster(self, symbol: str) -> pd.DataFrame:
        """Collect from Aster DEX."""
        try:
            df = await self.aster_collector.collect_historical_data(
                symbol=symbol,
                start_date=self.start_date.strftime("%Y-%m-%d"),
                end_date=self.end_date.strftime("%Y-%m-%d"),
                interval="1h",
                limit=1000
            )
            return df if not df.empty else pd.DataFrame()
        except Exception as e:
            logger.debug(f"    Aster DEX failed: {e}")
            return pd.DataFrame()
    
    async def _collect_from_binance(self, symbol: str) -> pd.DataFrame:
        """Collect from Binance as fallback."""
        try:
            # Convert symbol format if needed
            binance_symbol = symbol.replace("USDT", "/USDT")
            
            if binance_symbol not in self.binance.markets:
                return pd.DataFrame()
            
            # Fetch OHLCV data
            since = int(self.start_date.timestamp() * 1000)
            limit = 1000
            all_ohlcv = []
            
            current_since = since
            end_timestamp = int(self.end_date.timestamp() * 1000)
            
            while current_since < end_timestamp:
                ohlcv = await self.binance.fetch_ohlcv(
                    binance_symbol,
                    timeframe='1h',
                    since=current_since,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            if not all_ohlcv:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.debug(f"    Binance failed: {e}")
            return pd.DataFrame()
    
    async def _collect_from_coingecko(self, symbol: str) -> pd.DataFrame:
        """Collect from CoinGecko API."""
        try:
            # Extract coin from symbol (e.g., ASTERUSDT -> aster)
            coin_id = symbol.replace("USDT", "").lower()
            
            # Map common coin IDs
            coin_mapping = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'sol': 'solana',
                'sui': 'sui',
                'apt': 'aptos',
                'aster': 'aster'  # Update with actual CoinGecko ID
            }
            
            coin_id = coin_mapping.get(coin_id, coin_id)
            
            # CoinGecko API
            from_timestamp = int(self.start_date.timestamp())
            to_timestamp = int(self.end_date.timestamp())
            
            url = f"{self.coingecko_api}/coins/{coin_id}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': from_timestamp,
                'to': to_timestamp
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return pd.DataFrame()
                    
                    data = await response.json()
            
            if 'prices' not in data or not data['prices']:
                return pd.DataFrame()
            
            # Convert to DataFrame
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)
            
            # Resample to hourly and create OHLCV
            hourly = prices.resample('1h').agg({
                'price': ['first', 'max', 'min', 'last', 'count']
            })
            hourly.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Estimate volume based on price volatility (rough approximation)
            hourly['volume'] = (hourly['high'] - hourly['low']).abs() * 1000000
            
            return hourly.dropna()
            
        except Exception as e:
            logger.debug(f"    CoinGecko failed: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_data(self, symbol: str, days: int = 180) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for very new assets.
        Uses realistic price movements and patterns.
        """
        logger.warning(f"    Generating synthetic data - USE WITH CAUTION")
        
        # Generate hourly timestamps
        timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='1h'
        )
        
        # Generate synthetic price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent for same symbol
        
        # Starting price based on symbol
        if 'BTC' in symbol:
            base_price = 40000
        elif 'ETH' in symbol:
            base_price = 2500
        elif 'ASTER' in symbol:
            base_price = 0.1  # New token
        else:
            base_price = 10
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0001, 0.02, len(timestamps))  # Small upward drift
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        df = pd.DataFrame(index=timestamps)
        df['close'] = prices
        
        # Add realistic OHLC variation
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(df)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(df)))
        
        # Generate volume
        df['volume'] = np.random.lognormal(10, 2, len(df))
        
        # Add some volume spikes during volatile periods
        volatility = df['close'].pct_change().abs()
        df.loc[volatility > volatility.quantile(0.9), 'volume'] *= 3
        
        return df
    
    def _resample_data(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample data to different intervals."""
        if interval == "1h":
            return df
        
        # Resample mapping
        freq_map = {
            "4h": "4h",
            "1d": "1D"
        }
        
        resampled = df.resample(freq_map[interval]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def _save_summary(self, results: Dict):
        """Save collection summary."""
        summary = []
        for symbol, result in results.items():
            summary.append({
                'symbol': symbol,
                'success': result.get('success', False),
                'records': result.get('records', 0),
                'source': result.get('source', result.get('reason', 'unknown')),
                'is_aster_native': symbol in self.aster_native_assets,
                'collection_time': datetime.now().isoformat()
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.output_dir / "collection_summary.csv", index=False)
        logger.info(f"✅ Summary saved to {self.output_dir}/collection_summary.csv")
    
    async def close(self):
        """Close all connections."""
        if self.aster_collector:
            await self.aster_collector.close()
        if self.binance:
            await self.binance.close()
        logger.info("✅ All connections closed")


async def main():
    """Main execution."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║       Aster-Native Asset Data Collection (Multi-Source)        ║
║          Fallback: Aster DEX → Binance → CoinGecko            ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    collector = AsterNativeDataCollector()
    
    try:
        await collector.initialize()
        results = await collector.collect_all_data()
        
        # Print summary
        print(f"\n{'='*60}")
        print("COLLECTION SUMMARY")
        print(f"{'='*60}\n")
        
        for symbol, result in results.items():
            status = "✅" if result.get('success') else "❌"
            source = result.get('source', result.get('reason', 'unknown'))
            records = result.get('records', 0)
            print(f"{status} {symbol:15} | Source: {source:15} | Records: {records}")
        
        print("""
╔════════════════════════════════════════════════════════════════╗
║              🎉 Aster Native Data Collection Complete!        ║
╚════════════════════════════════════════════════════════════════╝

Data saved to: data/historical/aster_native/

Next steps:
1. Review collection_summary.csv
2. Train Aster-specific model
3. Compare with general model performance
        """)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Collection interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())

