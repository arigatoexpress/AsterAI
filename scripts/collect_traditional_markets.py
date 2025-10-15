#!/usr/bin/env python3
"""
Traditional Markets Data Collection
Collects S&P 500, commodities, and economic indicators data.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
import json
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import fredapi

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.api_manager import APIKeyManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MarketAsset:
    """Market asset configuration"""
    symbol: str
    name: str
    asset_type: str  # 'equity', 'commodity', 'index', 'economic'
    data_source: str
    yahoo_ticker: Optional[str] = None
    fred_series: Optional[str] = None


class TraditionalMarketsCollector:
    """
    Collects traditional market data including:
    - S&P 500 components
    - Major market indices
    - Commodities (Gold, Silver, Oil, etc.)
    - Economic indicators (GDP, CPI, rates, etc.)
    """
    
    def __init__(self, output_dir: str = "data/historical/traditional_markets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API manager
        self.api_manager = APIKeyManager()
        self.api_manager.load_credentials()
        
        # Initialize FRED API if key available
        self.fred = None
        if self.api_manager.credentials.fred_api_key:
            self.fred = fredapi.Fred(api_key=self.api_manager.credentials.fred_api_key)
        
        # Asset definitions
        self.assets = self._define_assets()
        
        # Collection settings
        self.start_date = datetime.now() - timedelta(days=730)  # 2 years
        self.end_date = datetime.now()
        
        # Alpha Vantage settings
        self.alpha_vantage_key = self.api_manager.credentials.alpha_vantage_key
        self.av_rate_limit = 5  # 5 calls per minute for free tier
        
        self.session = None
    
    def _define_assets(self) -> List[MarketAsset]:
        """Define all assets to collect"""
        assets = []
        
        # Major indices
        indices = [
            MarketAsset("SPY", "SPDR S&P 500 ETF", "index", "yahoo", yahoo_ticker="SPY"),
            MarketAsset("QQQ", "Invesco QQQ Trust", "index", "yahoo", yahoo_ticker="QQQ"),
            MarketAsset("DIA", "SPDR Dow Jones", "index", "yahoo", yahoo_ticker="DIA"),
            MarketAsset("IWM", "iShares Russell 2000", "index", "yahoo", yahoo_ticker="IWM"),
            MarketAsset("VTI", "Vanguard Total Market", "index", "yahoo", yahoo_ticker="VTI"),
            MarketAsset("EFA", "iShares MSCI EAFE", "index", "yahoo", yahoo_ticker="EFA"),
            MarketAsset("EEM", "iShares MSCI EM", "index", "yahoo", yahoo_ticker="EEM"),
            MarketAsset("VIX", "Volatility Index", "index", "yahoo", yahoo_ticker="^VIX")
        ]
        
        # Top S&P 500 components (Magnificent 7 + others)
        sp500_stocks = [
            MarketAsset("AAPL", "Apple Inc.", "equity", "yahoo", yahoo_ticker="AAPL"),
            MarketAsset("MSFT", "Microsoft Corp.", "equity", "yahoo", yahoo_ticker="MSFT"),
            MarketAsset("GOOGL", "Alphabet Inc.", "equity", "yahoo", yahoo_ticker="GOOGL"),
            MarketAsset("AMZN", "Amazon.com Inc.", "equity", "yahoo", yahoo_ticker="AMZN"),
            MarketAsset("NVDA", "NVIDIA Corp.", "equity", "yahoo", yahoo_ticker="NVDA"),
            MarketAsset("META", "Meta Platforms", "equity", "yahoo", yahoo_ticker="META"),
            MarketAsset("TSLA", "Tesla Inc.", "equity", "yahoo", yahoo_ticker="TSLA"),
            MarketAsset("BRK-B", "Berkshire Hathaway", "equity", "yahoo", yahoo_ticker="BRK-B"),
            MarketAsset("JPM", "JPMorgan Chase", "equity", "yahoo", yahoo_ticker="JPM"),
            MarketAsset("JNJ", "Johnson & Johnson", "equity", "yahoo", yahoo_ticker="JNJ"),
            MarketAsset("V", "Visa Inc.", "equity", "yahoo", yahoo_ticker="V"),
            MarketAsset("PG", "Procter & Gamble", "equity", "yahoo", yahoo_ticker="PG"),
            MarketAsset("MA", "Mastercard Inc.", "equity", "yahoo", yahoo_ticker="MA"),
            MarketAsset("HD", "Home Depot", "equity", "yahoo", yahoo_ticker="HD"),
            MarketAsset("UNH", "UnitedHealth", "equity", "yahoo", yahoo_ticker="UNH"),
            MarketAsset("DIS", "Walt Disney", "equity", "yahoo", yahoo_ticker="DIS"),
            MarketAsset("BAC", "Bank of America", "equity", "yahoo", yahoo_ticker="BAC"),
            MarketAsset("XOM", "Exxon Mobil", "equity", "yahoo", yahoo_ticker="XOM"),
            MarketAsset("CVX", "Chevron Corp.", "equity", "yahoo", yahoo_ticker="CVX"),
            MarketAsset("WMT", "Walmart Inc.", "equity", "yahoo", yahoo_ticker="WMT")
        ]
        
        # Commodities
        commodities = [
            MarketAsset("GLD", "SPDR Gold Trust", "commodity", "yahoo", yahoo_ticker="GLD"),
            MarketAsset("SLV", "iShares Silver Trust", "commodity", "yahoo", yahoo_ticker="SLV"),
            MarketAsset("USO", "United States Oil Fund", "commodity", "yahoo", yahoo_ticker="USO"),
            MarketAsset("UNG", "United States Natural Gas", "commodity", "yahoo", yahoo_ticker="UNG"),
            MarketAsset("DBA", "Agriculture Fund", "commodity", "yahoo", yahoo_ticker="DBA"),
            MarketAsset("DBB", "Base Metals Fund", "commodity", "yahoo", yahoo_ticker="DBB"),
            MarketAsset("URA", "Uranium ETF", "commodity", "yahoo", yahoo_ticker="URA"),
            MarketAsset("COPX", "Copper Miners ETF", "commodity", "yahoo", yahoo_ticker="COPX"),
            MarketAsset("LIT", "Lithium & Battery Tech", "commodity", "yahoo", yahoo_ticker="LIT"),
            MarketAsset("PALL", "Palladium ETF", "commodity", "yahoo", yahoo_ticker="PALL")
        ]
        
        # Economic indicators (FRED series)
        economic_indicators = [
            MarketAsset("GDP", "US GDP Growth", "economic", "fred", fred_series="GDP"),
            MarketAsset("CPIAUCSL", "Consumer Price Index", "economic", "fred", fred_series="CPIAUCSL"),
            MarketAsset("CPILFESL", "Core CPI", "economic", "fred", fred_series="CPILFESL"),
            MarketAsset("UNRATE", "Unemployment Rate", "economic", "fred", fred_series="UNRATE"),
            MarketAsset("DFF", "Fed Funds Rate", "economic", "fred", fred_series="DFF"),
            MarketAsset("DGS10", "10-Year Treasury", "economic", "fred", fred_series="DGS10"),
            MarketAsset("DGS2", "2-Year Treasury", "economic", "fred", fred_series="DGS2"),
            MarketAsset("T10Y2Y", "10Y-2Y Spread", "economic", "fred", fred_series="T10Y2Y"),
            MarketAsset("DEXUSEU", "USD/EUR Exchange", "economic", "fred", fred_series="DEXUSEU"),
            MarketAsset("M2SL", "M2 Money Supply", "economic", "fred", fred_series="M2SL"),
            MarketAsset("UMCSENT", "Consumer Sentiment", "economic", "fred", fred_series="UMCSENT"),
            MarketAsset("HOUST", "Housing Starts", "economic", "fred", fred_series="HOUST"),
            MarketAsset("INDPRO", "Industrial Production", "economic", "fred", fred_series="INDPRO"),
            MarketAsset("PAYEMS", "Nonfarm Payrolls", "economic", "fred", fred_series="PAYEMS"),
            MarketAsset("WALCL", "Fed Balance Sheet", "economic", "fred", fred_series="WALCL")
        ]
        
        # Treasury and bond data
        bonds = [
            MarketAsset("TLT", "20+ Year Treasury", "index", "yahoo", yahoo_ticker="TLT"),
            MarketAsset("IEF", "7-10 Year Treasury", "index", "yahoo", yahoo_ticker="IEF"),
            MarketAsset("SHY", "1-3 Year Treasury", "index", "yahoo", yahoo_ticker="SHY"),
            MarketAsset("AGG", "Aggregate Bond", "index", "yahoo", yahoo_ticker="AGG"),
            MarketAsset("HYG", "High Yield Bond", "index", "yahoo", yahoo_ticker="HYG"),
            MarketAsset("LQD", "Investment Grade Bond", "index", "yahoo", yahoo_ticker="LQD")
        ]
        
        # Combine all assets
        assets.extend(indices)
        assets.extend(sp500_stocks)
        assets.extend(commodities)
        assets.extend(economic_indicators)
        assets.extend(bonds)
        
        return assets
    
    async def initialize(self):
        """Initialize collectors"""
        logger.info("Initializing traditional markets collectors...")
        self.session = aiohttp.ClientSession()
        logger.info("✅ Collectors initialized")
    
    async def collect_all_assets(self) -> Dict[str, Dict]:
        """Collect data for all defined assets"""
        logger.info(f"Starting collection for {len(self.assets)} assets...")
        
        results = {}
        failed_assets = []
        
        # Group by data source for efficient collection
        yahoo_assets = [a for a in self.assets if a.data_source == "yahoo"]
        fred_assets = [a for a in self.assets if a.data_source == "fred"]
        alpha_vantage_assets = [a for a in self.assets if a.data_source == "alpha_vantage"]
        
        # Collect Yahoo Finance data
        logger.info(f"Collecting {len(yahoo_assets)} assets from Yahoo Finance...")
        yahoo_results = await self._collect_yahoo_batch(yahoo_assets)
        results.update(yahoo_results)
        
        # Collect FRED data
        if self.fred and fred_assets:
            logger.info(f"Collecting {len(fred_assets)} indicators from FRED...")
            fred_results = await self._collect_fred_batch(fred_assets)
            results.update(fred_results)
        
        # Collect Alpha Vantage data
        if self.alpha_vantage_key and alpha_vantage_assets:
            logger.info(f"Collecting {len(alpha_vantage_assets)} assets from Alpha Vantage...")
            av_results = await self._collect_alpha_vantage_batch(alpha_vantage_assets)
            results.update(av_results)
        
        # Identify failed assets
        for asset in self.assets:
            if asset.symbol not in results:
                failed_assets.append(asset.symbol)
        
        # Summary
        logger.info(f"""
Collection Summary:
==================
Total Assets: {len(self.assets)}
Successful: {len(results)}
Failed: {len(failed_assets)}
Success Rate: {len(results)/len(self.assets)*100:.1f}%
        """)
        
        # Save summary
        self._save_collection_summary(results, failed_assets)
        
        return results
    
    async def _collect_yahoo_batch(self, assets: List[MarketAsset]) -> Dict[str, Dict]:
        """Collect batch of assets from Yahoo Finance"""
        results = {}
        
        # Process in smaller batches
        batch_size = 10
        for i in range(0, len(assets), batch_size):
            batch = assets[i:i+batch_size]
            tickers = [a.yahoo_ticker for a in batch]
            
            try:
                # Use thread pool for yfinance
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    data = await loop.run_in_executor(
                        pool,
                        self._yahoo_download_batch,
                        tickers
                    )
                
                for asset, ticker in zip(batch, tickers):
                    if ticker in data and not data[ticker].empty:
                        asset_data = self._process_yahoo_data(asset, data[ticker])
                        results[asset.symbol] = asset_data
                        self._save_asset_data(asset.symbol, asset_data)
                        logger.info(f"✅ Collected {asset.symbol}: {len(asset_data['data'])} records")
                    
            except Exception as e:
                logger.error(f"Yahoo batch error: {e}")
            
            # Brief pause between batches
            await asyncio.sleep(1)
        
        return results
    
    def _yahoo_download_batch(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Download batch of tickers from Yahoo Finance"""
        try:
            data = yf.download(
                tickers,
                start=self.start_date,
                end=self.end_date,
                interval='1h',
                group_by='ticker',
                progress=False,
                threads=True
            )
            
            # Handle single ticker case
            if len(tickers) == 1:
                return {tickers[0]: data}
            
            # Split multi-ticker DataFrame
            result = {}
            for ticker in tickers:
                if ticker in data.columns.levels[0]:
                    result[ticker] = data[ticker]
            
            return result
            
        except Exception as e:
            logger.error(f"Yahoo download error: {e}")
            return {}
    
    def _process_yahoo_data(self, asset: MarketAsset, data: pd.DataFrame) -> Dict:
        """Process Yahoo Finance data"""
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in data.columns]
        
        data = data[available_cols].copy()
        
        # Fill missing volume for indices/commodities
        if 'volume' not in data.columns:
            data['volume'] = 0
        
        # Forward fill gaps (max 24 hours)
        data = data.fillna(method='ffill', limit=24)
        
        return {
            'symbol': asset.symbol,
            'name': asset.name,
            'asset_type': asset.asset_type,
            'data': data,
            'source': 'yahoo',
            'records': len(data),
            'date_range': f"{data.index.min()} to {data.index.max()}"
        }
    
    async def _collect_fred_batch(self, assets: List[MarketAsset]) -> Dict[str, Dict]:
        """Collect batch of indicators from FRED"""
        results = {}
        
        for asset in assets:
            try:
                # Fetch series data
                data = self.fred.get_series(
                    asset.fred_series,
                    observation_start=self.start_date,
                    observation_end=self.end_date
                )
                
                if data is not None and not data.empty:
                    # Convert to DataFrame with standard format
                    df = pd.DataFrame(data, columns=['value'])
                    
                    # Resample to daily frequency if needed
                    df = df.resample('D').last().fillna(method='ffill')
                    
                    # Create OHLCV format for consistency
                    df['open'] = df['value']
                    df['high'] = df['value']
                    df['low'] = df['value']
                    df['close'] = df['value']
                    df['volume'] = 0
                    
                    asset_data = {
                        'symbol': asset.symbol,
                        'name': asset.name,
                        'asset_type': asset.asset_type,
                        'data': df[['open', 'high', 'low', 'close', 'volume']],
                        'source': 'fred',
                        'records': len(df),
                        'date_range': f"{df.index.min()} to {df.index.max()}"
                    }
                    
                    results[asset.symbol] = asset_data
                    self._save_asset_data(asset.symbol, asset_data)
                    logger.info(f"✅ Collected {asset.symbol}: {len(df)} records")
                    
            except Exception as e:
                logger.error(f"FRED error for {asset.symbol}: {e}")
        
        return results
    
    async def _collect_alpha_vantage_batch(self, assets: List[MarketAsset]) -> Dict[str, Dict]:
        """Collect batch of assets from Alpha Vantage"""
        results = {}
        
        for asset in assets:
            try:
                # TIME_SERIES_INTRADAY endpoint
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_INTRADAY',
                    'symbol': asset.symbol,
                    'interval': '60min',
                    'outputsize': 'full',
                    'apikey': self.alpha_vantage_key
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'Time Series (60min)' in data:
                            # Convert to DataFrame
                            ts_data = data['Time Series (60min)']
                            df = pd.DataFrame.from_dict(ts_data, orient='index')
                            df.index = pd.to_datetime(df.index)
                            
                            # Rename columns
                            df.columns = ['open', 'high', 'low', 'close', 'volume']
                            df = df.astype(float)
                            df = df.sort_index()
                            
                            asset_data = {
                                'symbol': asset.symbol,
                                'name': asset.name,
                                'asset_type': asset.asset_type,
                                'data': df,
                                'source': 'alpha_vantage',
                                'records': len(df),
                                'date_range': f"{df.index.min()} to {df.index.max()}"
                            }
                            
                            results[asset.symbol] = asset_data
                            self._save_asset_data(asset.symbol, asset_data)
                            logger.info(f"✅ Collected {asset.symbol}: {len(df)} records")
                
                # Rate limiting (5 calls per minute for free tier)
                await asyncio.sleep(12)
                
            except Exception as e:
                logger.error(f"Alpha Vantage error for {asset.symbol}: {e}")
        
        return results
    
    def _save_asset_data(self, symbol: str, asset_data: Dict):
        """Save collected data to disk"""
        # Create subdirectory by asset type
        asset_type = asset_data['asset_type']
        type_dir = self.output_dir / asset_type
        type_dir.mkdir(exist_ok=True)
        
        # Save data as Parquet
        output_file = type_dir / f"{symbol}.parquet"
        asset_data['data'].to_parquet(output_file, compression='snappy')
        
        # Save metadata
        metadata = {
            'symbol': asset_data['symbol'],
            'name': asset_data['name'],
            'asset_type': asset_data['asset_type'],
            'source': asset_data['source'],
            'records': asset_data['records'],
            'date_range': asset_data['date_range'],
            'collection_time': datetime.now().isoformat()
        }
        
        metadata_file = type_dir / f"{symbol}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_collection_summary(self, results: Dict, failed_assets: List[str]):
        """Save overall collection summary"""
        # Group by asset type
        by_type = {}
        for symbol, data in results.items():
            asset_type = data['asset_type']
            if asset_type not in by_type:
                by_type[asset_type] = []
            by_type[asset_type].append({
                'symbol': symbol,
                'name': data['name'],
                'records': data['records'],
                'source': data['source']
            })
        
        summary = {
            'collection_time': datetime.now().isoformat(),
            'total_assets': len(self.assets),
            'successful': len(results),
            'failed': len(failed_assets),
            'failed_assets': failed_assets,
            'by_type': by_type,
            'sources_used': {
                'yahoo': len([r for r in results.values() if r['source'] == 'yahoo']),
                'fred': len([r for r in results.values() if r['source'] == 'fred']),
                'alpha_vantage': len([r for r in results.values() if r['source'] == 'alpha_vantage'])
            }
        }
        
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Collection summary saved to {summary_file}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()


async def main():
    """Main execution"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║         Traditional Markets Data Collection                    ║
║        S&P 500, Commodities, Economic Indicators              ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    collector = TraditionalMarketsCollector()
    
    try:
        await collector.initialize()
        results = await collector.collect_all_assets()
        
        print(f"\n✅ Collection complete!")
        print(f"   Data saved to: {collector.output_dir}")
        
    finally:
        await collector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

