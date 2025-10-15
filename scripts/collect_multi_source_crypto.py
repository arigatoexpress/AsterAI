#!/usr/bin/env python3
"""
Multi-Source Cryptocurrency Data Collection
Collects data from multiple sources for the top 100 cryptocurrencies.
Implements fallback mechanisms and data harmonization.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import aiohttp
import json
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import ccxt.async_support as ccxt
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.api_manager import APIKeyManager
from local_training.aster_dex_data_collector import AsterDEXDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    priority: int
    rate_limit: int  # requests per minute
    has_historical: bool
    requires_api_key: bool


class MultiSourceCryptoCollector:
    """
    Collects cryptocurrency data from multiple sources.
    Prioritizes real market data and implements fallback mechanisms.
    """
    
    def __init__(self, output_dir: str = "data/historical/multi_source"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API manager
        self.api_manager = APIKeyManager()
        self.api_manager.load_credentials()
        
        # Data sources configuration
        self.sources = {
            'aster': DataSource('aster', priority=1, rate_limit=300, has_historical=True, requires_api_key=True),
            'binance': DataSource('binance', priority=2, rate_limit=1200, has_historical=True, requires_api_key=False),
            'coingecko': DataSource('coingecko', priority=3, rate_limit=30, has_historical=True, requires_api_key=False),
            'cryptocompare': DataSource('cryptocompare', priority=4, rate_limit=100, has_historical=True, requires_api_key=False),
            'yahoo': DataSource('yahoo', priority=5, rate_limit=2000, has_historical=True, requires_api_key=False)
        }
        
        # Initialize collectors
        self.aster_collector = None
        self.binance_exchange = None
        self.session = None
        
        # Top 100 cryptocurrency symbols (by market cap)
        self.top_100_symbols = self._get_top_100_symbols()
        
        # Collection settings
        self.start_date = datetime.now() - timedelta(days=730)  # 2 years
        self.end_date = datetime.now()
        
        # Rate limiting
        self.rate_limiters = {source: RateLimiter(config.rate_limit) for source, config in self.sources.items()}
        
    def _get_top_100_symbols(self) -> List[str]:
        """Get top 100 cryptocurrency symbols by market cap"""
        # Core top 100 cryptocurrencies (as of 2024)
        return [
            "BTC", "ETH", "USDT", "BNB", "SOL", "XRP", "USDC", "ADA", "AVAX", "DOGE",
            "TRX", "DOT", "LINK", "MATIC", "TON", "DAI", "WBTC", "SHIB", "LTC", "BCH",
            "LEO", "UNI", "OKB", "ATOM", "XLM", "ETC", "HBAR", "CRO", "APT", "MKR",
            "VET", "NEAR", "GRT", "AAVE", "ALGO", "QNT", "FIL", "ICP", "EOS", "XTZ",
            "SAND", "MANA", "THETA", "AXS", "EGLD", "BSV", "XMR", "KCS", "TUSD", "GALA",
            "CHZ", "FLOW", "HT", "MINA", "KDA", "CRV", "XEC", "FTM", "GT", "KLAY",
            "ZEC", "IOTA", "NEO", "CAKE", "FXS", "ENJ", "COMP", "SNX", "ROSE", "LRC",
            "LUNA", "KAVA", "BAT", "DASH", "WAVES", "KSM", "ZIL", "CELO", "1INCH", "ANKR",
            "YFI", "RVN", "WOO", "GLM", "ICX", "BTG", "TWT", "CEL", "XEM", "QTUM",
            "OMG", "SXP", "ZRX", "ONT", "IOTX", "AUDIO", "STORJ", "RSR", "BAND", "OCEAN"
        ]
    
    async def initialize(self):
        """Initialize all data collectors"""
        logger.info("Initializing multi-source collectors...")
        
        # Initialize AsterDEX collector if API key available
        if self.api_manager.credentials.aster_api_key:
            self.aster_collector = AsterDEXDataCollector()
            await self.aster_collector.initialize()
        
        # Initialize Binance
        self.binance_exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 50  # 50ms between requests
        })
        
        # Initialize aiohttp session
        self.session = aiohttp.ClientSession()
        
        logger.info("✅ All collectors initialized")
    
    async def collect_all_assets(self) -> Dict[str, Dict]:
        """Collect data for all top 100 cryptocurrencies"""
        logger.info(f"Starting collection for {len(self.top_100_symbols)} assets...")
        
        results = {}
        failed_assets = []
        
        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(self.top_100_symbols), batch_size):
            batch = self.top_100_symbols[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{len(self.top_100_symbols)//batch_size + 1}")
            
            tasks = [self.collect_asset_data(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to collect {symbol}: {result}")
                    failed_assets.append(symbol)
                else:
                    results[symbol] = result
            
            # Brief pause between batches
            await asyncio.sleep(2)
        
        # Summary
        logger.info(f"""
Collection Summary:
==================
Total Assets: {len(self.top_100_symbols)}
Successful: {len(results)}
Failed: {len(failed_assets)}
Success Rate: {len(results)/len(self.top_100_symbols)*100:.1f}%
        """)
        
        # Save summary
        self._save_collection_summary(results, failed_assets)
        
        return results
    
    async def collect_asset_data(self, symbol: str) -> Dict[str, Any]:
        """Collect data for a single asset from multiple sources"""
        logger.info(f"Collecting data for {symbol}...")
        
        asset_data = {
            'symbol': symbol,
            'sources': {},
            'consolidated': None,
            'metadata': {
                'collection_time': datetime.now().isoformat(),
                'data_points': 0,
                'sources_used': []
            }
        }
        
        # Try each source in priority order
        for source_name in sorted(self.sources.keys(), key=lambda x: self.sources[x].priority):
            try:
                logger.debug(f"  Trying {source_name} for {symbol}...")
                
                # Apply rate limiting
                await self.rate_limiters[source_name].acquire()
                
                # Collect from source
                if source_name == 'aster' and self.aster_collector:
                    data = await self._collect_from_aster(symbol)
                elif source_name == 'binance':
                    data = await self._collect_from_binance(symbol)
                elif source_name == 'coingecko':
                    data = await self._collect_from_coingecko(symbol)
                elif source_name == 'cryptocompare':
                    data = await self._collect_from_cryptocompare(symbol)
                elif source_name == 'yahoo':
                    data = await self._collect_from_yahoo(symbol)
                else:
                    continue
                
                if data is not None and not data.empty:
                    asset_data['sources'][source_name] = {
                        'data': data,
                        'records': len(data),
                        'date_range': f"{data.index.min()} to {data.index.max()}"
                    }
                    asset_data['metadata']['sources_used'].append(source_name)
                    logger.info(f"  ✅ Collected {len(data)} records from {source_name}")
                
            except Exception as e:
                logger.warning(f"  ❌ Failed to collect from {source_name}: {str(e)}")
                continue
        
        # Consolidate data from multiple sources
        if asset_data['metadata']['sources_used']:
            asset_data['consolidated'] = self._consolidate_data(asset_data['sources'])
            asset_data['metadata']['data_points'] = len(asset_data['consolidated'])
            
            # Save to disk
            self._save_asset_data(symbol, asset_data)
        
        return asset_data
    
    async def _collect_from_aster(self, symbol: str) -> Optional[pd.DataFrame]:
        """Collect from Aster DEX"""
        if not self.aster_collector:
            return None
        
        trading_symbol = f"{symbol}USDT"
        data = await self.aster_collector.collect_historical_data(
            trading_symbol, 
            start_date=self.start_date.strftime('%Y-%m-%d'),
            end_date=self.end_date.strftime('%Y-%m-%d'),
            interval='1h'
        )
        
        if data is not None and not data.empty:
            return data
        return None
    
    async def _collect_from_binance(self, symbol: str) -> Optional[pd.DataFrame]:
        """Collect from Binance"""
        try:
            trading_symbol = f"{symbol}/USDT"
            
            # Check if symbol exists
            markets = await self.binance_exchange.load_markets()
            if trading_symbol not in markets:
                return None
            
            # Fetch OHLCV data
            ohlcv = await self.binance_exchange.fetch_ohlcv(
                trading_symbol,
                timeframe='1h',
                since=int(self.start_date.timestamp() * 1000),
                limit=1000
            )
            
            if not ohlcv:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.debug(f"Binance error for {symbol}: {e}")
            return None
    
    async def _collect_from_coingecko(self, symbol: str) -> Optional[pd.DataFrame]:
        """Collect from CoinGecko (free, no API key required)"""
        try:
            # Map symbol to CoinGecko ID (simplified mapping)
            symbol_map = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
                'SOL': 'solana', 'XRP': 'ripple', 'USDT': 'tether',
                'ADA': 'cardano', 'AVAX': 'avalanche-2', 'DOGE': 'dogecoin',
                'DOT': 'polkadot', 'MATIC': 'matic-network', 'LINK': 'chainlink',
                'UNI': 'uniswap', 'LTC': 'litecoin', 'ATOM': 'cosmos',
                'NEAR': 'near', 'AAVE': 'aave', 'ALGO': 'algorand'
                # Add more mappings as needed
            }
            
            coin_id = symbol_map.get(symbol, symbol.lower())
            
            # CoinGecko free API endpoint for historical data
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '730',  # 2 years
                'interval': 'hourly'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract price data
                    prices = data.get('prices', [])
                    if not prices:
                        return None
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Add basic OHLCV structure (close-only for free tier)
                    df['open'] = df['close']
                    df['high'] = df['close'] * 1.001  # Approximate
                    df['low'] = df['close'] * 0.999   # Approximate
                    df['volume'] = 0  # Not available in free tier
                    
                    return df[['open', 'high', 'low', 'close', 'volume']]
                    
        except Exception as e:
            logger.debug(f"CoinGecko error for {symbol}: {e}")
            return None
    
    async def _collect_from_cryptocompare(self, symbol: str) -> Optional[pd.DataFrame]:
        """Collect from CryptoCompare"""
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/histohour"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': 2000,
                'toTs': int(self.end_date.timestamp())
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data['Response'] == 'Success':
                        df = pd.DataFrame(data['Data']['Data'])
                        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('timestamp', inplace=True)
                        df = df[['open', 'high', 'low', 'close', 'volumefrom']].rename(
                            columns={'volumefrom': 'volume'}
                        )
                        return df
                        
        except Exception as e:
            logger.debug(f"CryptoCompare error for {symbol}: {e}")
            return None
    
    async def _collect_from_yahoo(self, symbol: str) -> Optional[pd.DataFrame]:
        """Collect from Yahoo Finance (for crypto tickers)"""
        try:
            # Yahoo uses different naming convention
            ticker = f"{symbol}-USD"
            
            # Use yfinance in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                df = await loop.run_in_executor(
                    pool,
                    self._yahoo_download,
                    ticker,
                    self.start_date,
                    self.end_date
                )
            
            if df is not None and not df.empty:
                # Resample to hourly if needed
                df = df.resample('1H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                return df
                
        except Exception as e:
            logger.debug(f"Yahoo error for {symbol}: {e}")
            return None
    
    def _yahoo_download(self, ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Download data from Yahoo Finance (blocking)"""
        try:
            return yf.download(ticker, start=start, end=end, interval='1h', progress=False)
        except:
            return None
    
    def _consolidate_data(self, sources: Dict[str, Dict]) -> pd.DataFrame:
        """Consolidate data from multiple sources"""
        # Priority order for data consolidation
        priority_order = ['aster', 'binance', 'coingecko', 'cryptocompare', 'yahoo']
        
        consolidated = None
        
        for source in priority_order:
            if source in sources and sources[source]['data'] is not None:
                source_data = sources[source]['data']
                
                if consolidated is None:
                    consolidated = source_data.copy()
                else:
                    # Fill missing data with secondary sources
                    consolidated = consolidated.combine_first(source_data)
        
        if consolidated is not None:
            # Sort by timestamp
            consolidated = consolidated.sort_index()
            
            # Forward fill any remaining gaps (max 24 hours)
            consolidated = consolidated.fillna(method='ffill', limit=24)
            
            # Add metadata
            consolidated.attrs['sources'] = list(sources.keys())
            consolidated.attrs['consolidation_time'] = datetime.now().isoformat()
        
        return consolidated
    
    def _save_asset_data(self, symbol: str, asset_data: Dict):
        """Save collected data to disk"""
        # Save consolidated data as Parquet
        if asset_data['consolidated'] is not None:
            output_file = self.output_dir / f"{symbol}_consolidated.parquet"
            asset_data['consolidated'].to_parquet(output_file, compression='snappy')
            logger.info(f"Saved {symbol} data to {output_file}")
        
        # Save metadata
        metadata_file = self.output_dir / f"{symbol}_metadata.json"
        metadata = {
            'symbol': symbol,
            'sources_used': asset_data['metadata']['sources_used'],
            'data_points': asset_data['metadata']['data_points'],
            'collection_time': asset_data['metadata']['collection_time'],
            'source_details': {
                source: {
                    'records': details['records'],
                    'date_range': details['date_range']
                }
                for source, details in asset_data['sources'].items()
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_collection_summary(self, results: Dict, failed_assets: List[str]):
        """Save overall collection summary"""
        summary = {
            'collection_time': datetime.now().isoformat(),
            'total_assets': len(self.top_100_symbols),
            'successful': len(results),
            'failed': len(failed_assets),
            'failed_assets': failed_assets,
            'asset_summary': {}
        }
        
        for symbol, data in results.items():
            summary['asset_summary'][symbol] = {
                'sources': data['metadata']['sources_used'],
                'data_points': data['metadata']['data_points']
            }
        
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Collection summary saved to {summary_file}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        if self.binance_exchange:
            await self.binance_exchange.close()
        if self.aster_collector:
            await self.aster_collector.close()


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0
    
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        time_since_last = now - self.last_call
        
        if time_since_last < self.interval:
            await asyncio.sleep(self.interval - time_since_last)
        
        self.last_call = time.time()


async def main():
    """Main execution"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║        Multi-Source Cryptocurrency Data Collection             ║
║                  Top 100 Cryptocurrencies                      ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    collector = MultiSourceCryptoCollector()
    
    try:
        await collector.initialize()
        results = await collector.collect_all_assets()
        
        print(f"\n✅ Collection complete!")
        print(f"   Data saved to: {collector.output_dir}")
        
    finally:
        await collector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

