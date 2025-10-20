#!/usr/bin/env python3
"""
Real-Time Price Fetcher for Live Market Data
Fetches current cryptocurrency prices from multiple free APIs
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Optional, List, Any
import logging

logger = logging.getLogger(__name__)

class RealTimePriceFetcher:
    """Fetch real-time cryptocurrency prices from free APIs"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_update = {}
        self.cache = {}
        self.cache_timeout = 30  # seconds

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_coingecko_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get prices from CoinGecko (free, no API key needed)"""
        try:
            if not self.session:
                return {}

            # CoinGecko uses different symbol format
            coingecko_ids = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'SOL': 'solana',
                'ADA': 'cardano',
                'DOT': 'polkadot',
                'LINK': 'chainlink',
                'AVAX': 'avalanche-2',
                'MATIC': 'matic-network'
            }

            ids = [coingecko_ids.get(symbol.upper(), symbol.lower()) for symbol in symbols if symbol.upper() in coingecko_ids]

            if not ids:
                return {}

            url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"

            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    result = {}
                    for symbol, coin_id in coingecko_ids.items():
                        if coin_id in data:
                            price_data = data[coin_id]
                            result[symbol] = {
                                'price': price_data.get('usd', 0),
                                'change_24h': price_data.get('usd_24h_change', 0),
                                'volume_24h': price_data.get('usd_24h_vol', 0),
                                'source': 'coingecko',
                                'timestamp': datetime.now().isoformat()
                            }

                    return result
                else:
                    logger.warning(f"CoinGecko API error: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error fetching CoinGecko prices: {e}")
            return {}

    async def get_binance_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get prices from Binance (free, no API key needed)"""
        try:
            if not self.session:
                return {}

            # Binance uses USDT pairs
            symbols_usdt = [f"{symbol.upper()}USDT" for symbol in symbols]

            url = "https://api.binance.com/api/v3/ticker/24hr"

            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    result = {}
                    for item in data:
                        symbol = item['symbol']
                        if symbol in symbols_usdt:
                            base_symbol = symbol.replace('USDT', '')
                            result[base_symbol] = {
                                'price': float(item.get('lastPrice', 0)),
                                'change_24h': float(item.get('priceChangePercent', 0)),
                                'volume_24h': float(item.get('volume', 0)),
                                'source': 'binance',
                                'timestamp': datetime.now().isoformat()
                            }

                    return result
                else:
                    logger.warning(f"Binance API error: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error fetching Binance prices: {e}")
            return {}

    async def get_yahoo_finance_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get prices from Yahoo Finance (free, no API key needed)"""
        try:
            import yfinance as yf

            result = {}
            for symbol in symbols:
                try:
                    # For crypto, Yahoo Finance uses different symbols
                    yahoo_symbol = f"{symbol.upper()}-USD" if symbol.upper() in ['BTC', 'ETH', 'SOL', 'ADA'] else symbol

                    ticker = yf.Ticker(yahoo_symbol)
                    info = ticker.info

                    if info and 'regularMarketPrice' in info:
                        # Get 24h change if available
                        hist = ticker.history(period="1d")
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            prev_close = hist['Open'].iloc[0] if len(hist) > 1 else current_price
                            change_24h = ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                        else:
                            current_price = info.get('regularMarketPrice', 0)
                            change_24h = 0

                        result[symbol] = {
                            'price': float(current_price),
                            'change_24h': float(change_24h),
                            'volume_24h': float(info.get('volume24Hr', 0)),
                            'source': 'yahoo_finance',
                            'timestamp': datetime.now().isoformat()
                        }

                except Exception as e:
                    logger.warning(f"Error fetching {symbol} from Yahoo Finance: {e}")
                    continue

            return result

        except ImportError:
            logger.warning("yfinance not available, skipping Yahoo Finance")
            return {}
        except Exception as e:
            logger.error(f"Error with Yahoo Finance: {e}")
            return {}

    async def get_current_prices(self, symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get current prices from all available sources"""
        if symbols is None:
            symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK']

        all_prices = {}

        # Try multiple sources in order of preference
        sources = [
            ('binance', self.get_binance_prices),
            ('coingecko', self.get_coingecko_prices),
            ('yahoo_finance', self.get_yahoo_finance_prices)
        ]

        for source_name, fetch_func in sources:
            try:
                prices = await fetch_func(symbols)
                if prices:
                    logger.info(f"Got {len(prices)} prices from {source_name}")
                    # Merge with existing data, preferring newer sources
                    for symbol, data in prices.items():
                        all_prices[symbol] = data

            except Exception as e:
                logger.warning(f"Failed to get prices from {source_name}: {e}")

        # Update cache
        self.cache = all_prices
        self.last_update = {symbol: datetime.now() for symbol in all_prices.keys()}

        return all_prices

    def get_cached_prices(self) -> Dict[str, Dict[str, Any]]:
        """Get cached prices if still valid"""
        current_time = datetime.now()

        # Filter out expired cache entries
        valid_cache = {}
        for symbol, data in self.cache.items():
            if symbol in self.last_update:
                age = (current_time - self.last_update[symbol]).total_seconds()
                if age < self.cache_timeout:
                    valid_cache[symbol] = data

        return valid_cache

async def test_price_fetcher():
    """Test the price fetcher"""
    print("🧪 Testing Real-Time Price Fetcher...")

    async with RealTimePriceFetcher() as fetcher:
        print("📈 Fetching current prices...")

        prices = await fetcher.get_current_prices(['BTC', 'ETH', 'SOL'])

        if prices:
            print("✅ Successfully fetched prices:")
            for symbol, data in prices.items():
                price = data.get('price', 0)
                change = data.get('change_24h', 0)
                source = data.get('source', 'unknown')
                print(".2f")
        else:
            print("❌ No prices fetched")

    return prices

if __name__ == "__main__":
    # Run test
    asyncio.run(test_price_fetcher())
