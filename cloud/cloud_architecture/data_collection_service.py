#!/usr/bin/env python3
"""
Cloud Data Collection Service
Continuously collects data from Aster DEX for AI training
"""

import os
import asyncio
import logging
import json
from datetime import datetime, timedelta
import aiohttp
import google.cloud.storage as storage
from google.cloud import bigquery
import psutil
import time

# Configuration
ASTEX_API_BASE = "https://fapi.asterdex.com"
COLLECTION_INTERVAL = int(os.environ.get('COLLECTION_INTERVAL', 300))  # 5 minutes
GCP_PROJECT = os.environ.get('GCP_PROJECT', 'aster-ai-trading')
BUCKET_DATA = os.environ.get('BUCKET_DATA', 'aster-trading-data')
DATASET_ID = os.environ.get('DATASET_ID', 'trading_data')

# Symbols to collect
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'ADAUSDT', 'DOTUSDT', 'AVAXUSDT']

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudDataCollector:
    """Cloud-based data collection service."""

    def __init__(self):
        self.session = None
        self.storage_client = storage.Client(project=GCP_PROJECT)
        self.bq_client = bigquery.Client(project=GCP_PROJECT)
        self.bucket = self.storage_client.bucket(BUCKET_DATA)
        self.is_running = True
        self.collection_stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'data_points': 0
        }

    async def initialize(self):
        """Initialize HTTP session and clients."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        logger.info("Data collection service initialized")

    async def collect_market_data(self, symbol: str) -> dict:
        """Collect market data for a symbol."""
        try:
            # Get klines (OHLCV)
            kline_url = f"{ASTEX_API_BASE}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': '1h',
                'limit': 100
            }

            async with self.session.get(kline_url, params=params) as response:
                if response.status == 200:
                    klines = await response.json()
                else:
                    logger.warning(f"Failed to get klines for {symbol}: {response.status}")
                    return None

            # Get ticker (24hr stats)
            ticker_url = f"{ASTEX_API_BASE}/fapi/v1/ticker/24hr"
            async with self.session.get(ticker_url, params={'symbol': symbol}) as response:
                if response.status == 200:
                    ticker = await response.json()
                else:
                    ticker = {}

            # Get orderbook
            depth_url = f"{ASTEX_API_BASE}/fapi/v1/depth"
            async with self.session.get(depth_url, params={'symbol': symbol, 'limit': 20}) as response:
                if response.status == 200:
                    orderbook = await response.json()
                else:
                    orderbook = {}

            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'klines': klines,
                'ticker': ticker,
                'orderbook': orderbook
            }

        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, data: dict) -> dict:
        """Calculate technical indicators from raw data."""
        try:
            klines = data['klines']
            if not klines:
                return data

            # Extract OHLCV
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            # Calculate indicators
            indicators = {}

            # Simple Moving Averages
            if len(closes) >= 20:
                indicators['sma_5'] = sum(closes[-5:]) / 5
                indicators['sma_10'] = sum(closes[-10:]) / 10
                indicators['sma_20'] = sum(closes[-20:]) / 20

            # RSI
            if len(closes) >= 14:
                gains = []
                losses = []
                for i in range(1, len(closes)):
                    change = closes[i] - closes[i-1]
                    gains.append(max(change, 0))
                    losses.append(max(-change, 0))

                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14

                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    indicators['rsi'] = 100 - (100 / (1 + rs))

            # Volatility (20-period)
            if len(closes) >= 20:
                returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
                indicators['volatility'] = sum([r**2 for r in returns[-20:]]) / 20

            # Volume indicators
            if volumes:
                indicators['avg_volume'] = sum(volumes) / len(volumes)
                indicators['volume_trend'] = volumes[-1] / volumes[0] if volumes[0] > 0 else 1

            data['indicators'] = indicators
            return data

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return data

    async def save_to_cloud_storage(self, data: dict):
        """Save data to Google Cloud Storage."""
        try:
            symbol = data['symbol']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save raw data
            blob_name = f"raw/{symbol}/{timestamp}.json"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(json.dumps(data), content_type='application/json')

            # Save processed data with indicators
            processed_data = self.calculate_technical_indicators(data)
            processed_blob_name = f"processed/{symbol}/{timestamp}.json"
            processed_blob = self.bucket.blob(processed_blob_name)
            processed_blob.upload_from_string(json.dumps(processed_data), content_type='application/json')

            logger.info(f"Saved data for {symbol} to Cloud Storage")

        except Exception as e:
            logger.error(f"Error saving to Cloud Storage: {e}")

    async def save_to_bigquery(self, data: dict):
        """Save processed data to BigQuery."""
        try:
            symbol = data['symbol']
            indicators = data.get('indicators', {})

            # Prepare market data row
            market_row = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'price': float(data['ticker'].get('lastPrice', 0)) if data.get('ticker') else 0,
                'volume': float(data['ticker'].get('volume', 0)) if data.get('ticker') else 0,
                'volatility': indicators.get('volatility', 0)
            }

            # Insert to BigQuery
            table_id = f"{GCP_PROJECT}.{DATASET_ID}.market_data"
            table = self.bq_client.get_table(table_id)

            errors = self.bq_client.insert_rows_json(table, [market_row])
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Saved market data for {symbol} to BigQuery")

        except Exception as e:
            logger.error(f"Error saving to BigQuery: {e}")

    async def generate_daily_summary(self):
        """Generate daily data collection summary."""
        try:
            summary = {
                'date': datetime.now().date().isoformat(),
                'total_collections': self.collection_stats['total_collections'],
                'successful_collections': self.collection_stats['successful_collections'],
                'failed_collections': self.collection_stats['failed_collections'],
                'data_points': self.collection_stats['data_points'],
                'success_rate': (self.collection_stats['successful_collections'] /
                               max(self.collection_stats['total_collections'], 1)) * 100,
                'symbols_covered': SYMBOLS,
                'timestamp': datetime.now().isoformat()
            }

            # Save summary to Cloud Storage
            blob_name = f"summaries/daily/{summary['date']}.json"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(json.dumps(summary), content_type='application/json')

            logger.info(f"Daily summary generated: {summary['success_rate']:.1f}% success rate")

        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")

    async def run_collection_cycle(self):
        """Run one complete data collection cycle."""
        logger.info("Starting data collection cycle...")

        for symbol in SYMBOLS:
            try:
                # Collect data
                data = await self.collect_market_data(symbol)
                if not data:
                    self.collection_stats['failed_collections'] += 1
                    continue

                # Save to storage
                await self.save_to_cloud_storage(data)
                await self.save_to_bigquery(data)

                self.collection_stats['successful_collections'] += 1
                self.collection_stats['data_points'] += len(data.get('klines', []))

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self.collection_stats['failed_collections'] += 1

        self.collection_stats['total_collections'] += 1
        logger.info(f"Collection cycle complete: {self.collection_stats['successful_collections']}/{len(SYMBOLS)} symbols")

    async def run_service(self):
        """Main service loop."""
        await self.initialize()

        logger.info(f"Data collection service running (interval: {COLLECTION_INTERVAL}s)")

        while self.is_running:
            try:
                await self.run_collection_cycle()

                # Generate daily summary at 6 AM UTC
                now = datetime.now()
                if now.hour == 6 and now.minute < 5:  # Within first 5 minutes of 6 AM
                    await self.generate_daily_summary()

                await asyncio.sleep(COLLECTION_INTERVAL)

            except Exception as e:
                logger.error(f"Service error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    def get_status(self):
        """Get service status."""
        return {
            'service': 'data_collection',
            'running': self.is_running,
            'stats': self.collection_stats,
            'next_collection_in': COLLECTION_INTERVAL,
            'symbols': SYMBOLS,
            'cloud_storage': BUCKET_DATA,
            'bigquery_dataset': f"{GCP_PROJECT}.{DATASET_ID}"
        }

    async def shutdown(self):
        """Clean shutdown."""
        self.is_running = False
        if self.session:
            await self.session.close()
        logger.info("Data collection service shut down")

# Web server for health checks and API
from aiohttp import web

async def health_check(request):
    """Health check endpoint."""
    return web.json_response({
        'status': 'healthy',
        'service': 'data_collection',
        'timestamp': datetime.now().isoformat()
    })

async def status_endpoint(request):
    """Status endpoint."""
    collector = request.app['collector']
    return web.json_response(collector.get_status())

async def summarize_endpoint(request):
    """Manual summary generation."""
    collector = request.app['collector']
    await collector.generate_daily_summary()
    return web.json_response({'status': 'summary_generated'})

async def init_app():
    """Initialize web application."""
    app = web.Application()
    collector = CloudDataCollector()
    app['collector'] = collector

    # Routes
    app.router.add_get('/health', health_check)
    app.router.add_get('/status', status_endpoint)
    app.router.add_post('/summarize', summarize_endpoint)

    # Start background service
    asyncio.create_task(collector.run_service())

    return app

if __name__ == "__main__":
    # Run as web service
    app = asyncio.run(init_app())
    web.run_app(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
