#!/usr/bin/env python3
"""
Cloud Live Trading Service
Executes automated trades based on AI signals
"""

import os
import asyncio
import logging
import json
from datetime import datetime, timedelta
import aiohttp
import google.cloud.storage as storage
from google.cloud import bigquery
import joblib
import numpy as np
import pandas as pd

# Configuration
GCP_PROJECT = os.environ.get('GCP_PROJECT', 'aster-ai-trading')
BUCKET_MODELS = os.environ.get('BUCKET_MODELS', 'aster-trading-models')
DATASET_ID = os.environ.get('DATASET_ID', 'trading_data')
ASTEX_API_BASE = "https://fapi.asterdex.com"
MAX_LEVERAGE = int(os.environ.get('MAX_LEVERAGE', 20))
RISK_LIMIT = float(os.environ.get('RISK_LIMIT', 50))  # $50 daily loss limit

# Trading parameters
TRADE_SIZE_MIN = 10.0
TRADE_SIZE_MAX = 100.0
CONFIDENCE_THRESHOLD = 0.65
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.10

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudTradingService:
    """Cloud-based live trading service."""

    def __init__(self):
        self.storage_client = storage.Client(project=GCP_PROJECT)
        self.bq_client = bigquery.Client(project=GCP_PROJECT)
        self.bucket = self.storage_client.bucket(BUCKET_MODELS)
        self.session = None
        self.is_running = False

        # Trading state
        self.positions = []
        self.daily_pnl = 0
        self.total_pnl = 0
        self.capital = 150.0  # Starting capital
        self.trades_today = []
        self.last_trade = None

        # Load latest AI model
        self.model = None
        self.load_latest_model()

        logger.info("Live trading service initialized")

    async def initialize(self):
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

    def load_latest_model(self):
        """Load the latest trained model from Cloud Storage."""
        try:
            # Find latest model files
            blobs = list(self.bucket.list_blobs(prefix="models/"))
            if not blobs:
                logger.warning("No trained models found in Cloud Storage")
                return

            # Get latest ensemble model
            ensemble_blobs = [b for b in blobs if 'random_forest' in b.name]
            if not ensemble_blobs:
                logger.warning("No Random Forest model found")
                return

            # Sort by creation time (latest first)
            latest_blob = max(ensemble_blobs, key=lambda x: x.time_created)

            # Download and load model
            buffer = latest_blob.download_as_bytes()
            self.model = joblib.loads(buffer)

            logger.info(f"Loaded latest model: {latest_blob.name}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")

    async def get_market_data(self, symbol: str) -> dict:
        """Get current market data for a symbol."""
        try:
            # Get ticker data
            ticker_url = f"{ASTEX_API_BASE}/fapi/v1/ticker/price"
            async with self.session.get(ticker_url, params={'symbol': symbol}) as response:
                if response.status == 200:
                    ticker = await response.json()
                    price = float(ticker.get('price', 0))
                else:
                    logger.warning(f"Failed to get ticker for {symbol}")
                    return {}

            # Get recent klines for technical analysis
            kline_url = f"{ASTEX_API_BASE}/fapi/v1/klines"
            async with self.session.get(kline_url,
                                      params={'symbol': symbol, 'interval': '1h', 'limit': 50}) as response:
                if response.status == 200:
                    klines = await response.json()
                else:
                    klines = []

            return {
                'symbol': symbol,
                'price': price,
                'timestamp': datetime.now().isoformat(),
                'klines': klines
            }

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}

    def calculate_features(self, market_data: dict) -> np.ndarray:
        """Calculate features for AI model."""
        try:
            klines = market_data.get('klines', [])
            if not klines:
                return None

            # Extract OHLCV
            closes = [float(k[4]) for k in klines[-50:]]  # Last 50 candles
            volumes = [float(k[5]) for k in klines[-50:]]

            if len(closes) < 20:
                return None

            # Calculate features
            features = []

            # Price changes
            features.append((closes[-1] - closes[-2]) / closes[-2])  # price_change
            if len(closes) >= 5:
                features.append((closes[-1] - closes[-5]) / closes[-5])  # price_change_5
            else:
                features.append(0)

            if len(closes) >= 20:
                features.append((closes[-1] - closes[-20]) / closes[-20])  # price_change_20
            else:
                features.append(0)

            # Volume change
            if len(volumes) >= 2:
                features.append((volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0)
            else:
                features.append(0)

            # Volatility (20-period)
            if len(closes) >= 20:
                returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
                volatility = np.std(returns[-20:])
                features.append(volatility)
            else:
                features.append(0)

            # Moving averages
            sma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
            features.append(closes[-1] / sma_5 - 1)  # price_sma_5_ratio

            sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
            features.append(closes[-1] / sma_10 - 1)  # price_sma_10_ratio

            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
            features.append(closes[-1] / sma_20 - 1)  # price_sma_20_ratio

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None

    def generate_signal(self, market_data: dict) -> dict:
        """Generate trading signal using AI model."""
        try:
            if not self.model:
                return {'type': None, 'reason': 'No AI model loaded'}

            # Calculate features
            features = self.calculate_features(market_data)
            if features is None:
                return {'type': None, 'reason': 'Insufficient data'}

            # Get AI prediction
            prediction = self.model.predict(features)[0]
            confidence = max(self.model.predict_proba(features)[0])

            symbol = market_data['symbol']
            price = market_data['price']

            # Generate signal based on prediction and confidence
            if confidence >= CONFIDENCE_THRESHOLD:
                if prediction == 1:  # Long signal
                    return {
                        'type': 'scalping' if confidence < 0.75 else 'momentum',
                        'direction': 'long',
                        'symbol': symbol,
                        'entry_price': price,
                        'confidence': confidence,
                        'stop_loss': price * (1 - STOP_LOSS_PCT),
                        'take_profit': price * (1 + TAKE_PROFIT_PCT),
                        'reason': f'AI long signal ({confidence:.2f} confidence)'
                    }
                else:  # Short signal
                    return {
                        'type': 'scalping' if confidence < 0.75 else 'momentum',
                        'direction': 'short',
                        'symbol': symbol,
                        'entry_price': price,
                        'confidence': confidence,
                        'stop_loss': price * (1 + STOP_LOSS_PCT),
                        'take_profit': price * (1 - TAKE_PROFIT_PCT),
                        'reason': f'AI short signal ({confidence:.2f} confidence)'
                    }
            else:
                return {'type': None, 'reason': f'Low confidence ({confidence:.2f})'}

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {'type': None, 'reason': f'Error: {str(e)}'}

    def calculate_position_size(self, signal: dict) -> dict:
        """Calculate position size with leverage."""
        try:
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']

            # Risk per trade (1% of capital)
            risk_amount = self.capital * 0.01

            # Calculate position size based on stop loss
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit == 0:
                return None

            position_size = risk_amount / risk_per_unit

            # Apply leverage
            leverage = min(MAX_LEVERAGE, 10 if signal['type'] == 'scalping' else 5)
            notional_value = position_size * entry_price
            margin_required = notional_value / leverage

            # Check if we have enough capital
            available_capital = self.capital - sum(p.get('margin_required', 0) for p in self.positions)
            if margin_required > available_capital:
                # Scale down position
                margin_required = available_capital * 0.8  # Use 80% of available
                notional_value = margin_required * leverage
                position_size = notional_value / entry_price

            # Ensure within bounds
            position_size = max(TRADE_SIZE_MIN, min(TRADE_SIZE_MAX, position_size))

            return {
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'entry_price': entry_price,
                'position_size': position_size,
                'notional_value': notional_value,
                'margin_required': margin_required,
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit': signal['take_profit'],
                'confidence': signal['confidence'],
                'reason': signal['reason']
            }

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None

    async def execute_trade(self, position: dict):
        """Execute a trade (in production, this would call Aster DEX API)."""
        try:
            logger.info("🎯 EXECUTING TRADE")
            logger.info(f"   Symbol: {position['symbol']}")
            logger.info(f"   Direction: {position['direction']}")
            logger.info(f"   Size: ${position['position_size']:.2f}")
            logger.info(f"   Leverage: {position['leverage']}x")
            logger.info(f"   Entry: ${position['entry_price']:.2f}")
            logger.info(f"   Stop Loss: ${position['stop_loss']:.2f}")
            logger.info(f"   Take Profit: ${position['take_profit']:.2f}")
            logger.info(f"   Confidence: {position['confidence']:.2f}")
            logger.info(f"   Margin: ${position['margin_required']:.2f}")

            # In production, this would place the actual trade
            # For demo, we simulate the trade execution

            # Add to positions
            position['entry_time'] = datetime.now()
            self.positions.append(position)

            # Update capital
            self.capital -= position['margin_required']

            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': position['symbol'],
                'direction': position['direction'],
                'size': position['position_size'],
                'entry_price': position['entry_price'],
                'leverage': position['leverage'],
                'margin': position['margin_required'],
                'confidence': position['confidence'],
                'status': 'open'
            }
            self.trades_today.append(trade_record)
            self.last_trade = datetime.now()

            # Save trade to BigQuery
            await self.save_trade_to_bigquery(trade_record)

            logger.info("✅ Trade executed successfully")

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

    async def save_trade_to_bigquery(self, trade: dict):
        """Save trade to BigQuery."""
        try:
            row = {
                'timestamp': trade['timestamp'],
                'symbol': trade['symbol'],
                'direction': trade['direction'],
                'size': trade['size'],
                'entry_price': trade['entry_price'],
                'exit_price': None,  # Will be filled when closed
                'pnl': 0.0,  # Will be calculated when closed
                'confidence': trade['confidence']
            }

            table_id = f"{GCP_PROJECT}.{DATASET_ID}.trades"
            table = self.bq_client.get_table(table_id)

            errors = self.bq_client.insert_rows_json(table, [row])
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info("Trade saved to BigQuery")

        except Exception as e:
            logger.error(f"Error saving trade to BigQuery: {e}")

    async def manage_positions(self):
        """Monitor and manage open positions."""
        try:
            for position in self.positions[:]:  # Copy to avoid modification during iteration
                # Get current price (in production, this would be real-time)
                current_price = position['entry_price'] * (1 + np.random.normal(0, 0.02))  # Simulate price movement

                # Check stop loss
                if position['direction'] == 'long':
                    if current_price <= position['stop_loss']:
                        await self.close_position(position, current_price, "stop_loss")
                        continue
                    elif current_price >= position['take_profit']:
                        await self.close_position(position, current_price, "take_profit")
                        continue
                else:  # short
                    if current_price >= position['stop_loss']:
                        await self.close_position(position, current_price, "stop_loss")
                        continue
                    elif current_price <= position['take_profit']:
                        await self.close_position(position, current_price, "take_profit")
                        continue

                # Update unrealized P&L
                if position['direction'] == 'long':
                    pnl = (current_price - position['entry_price']) * position['position_size']
                else:
                    pnl = (position['entry_price'] - current_price) * position['position_size']

                position['unrealized_pnl'] = pnl

        except Exception as e:
            logger.error(f"Error managing positions: {e}")

    async def close_position(self, position: dict, exit_price: float, reason: str):
        """Close a position."""
        try:
            # Calculate P&L
            if position['direction'] == 'long':
                pnl = (exit_price - position['entry_price']) * position['position_size']
            else:
                pnl = (position['entry_price'] - exit_price) * position['position_size']

            # Apply leverage to P&L
            pnl *= position['leverage']

            logger.info(f"🔄 CLOSING POSITION: {position['symbol']} {position['direction']}")
            logger.info(f"   Entry: ${position['entry_price']:.2f}")
            logger.info(f"   Exit: ${exit_price:.2f}")
            logger.info(f"   P&L: ${pnl:.2f} ({pnl/position['margin_required']*100:.1f}%)")
            logger.info(f"   Reason: {reason}")

            # Update capital
            self.capital += position['margin_required'] + pnl
            self.daily_pnl += pnl
            self.total_pnl += pnl

            # Remove from positions
            self.positions.remove(position)

            # Update trade record
            for trade in self.trades_today:
                if (trade['symbol'] == position['symbol'] and
                    trade['direction'] == position['direction'] and
                    trade['entry_price'] == position['entry_price']):
                    trade['exit_price'] = exit_price
                    trade['pnl'] = pnl
                    trade['exit_time'] = datetime.now()
                    trade['status'] = 'closed'
                    trade['reason'] = reason

                    # Save updated trade to BigQuery
                    await self.save_trade_to_bigquery(trade)
                    break

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    async def scan_opportunities(self):
        """Scan for trading opportunities."""
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT']

        for symbol in symbols:
            try:
                # Get market data
                market_data = await self.get_market_data(symbol)
                if not market_data:
                    continue

                # Generate signal
                signal = self.generate_signal(market_data)
                if signal.get('type') is None:
                    continue

                # Calculate position size
                position = self.calculate_position_size(signal)
                if not position:
                    continue

                # Check if we should execute
                if len(self.positions) < 2 and self.daily_pnl > -RISK_LIMIT:
                    await self.execute_trade(position)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

    async def run_service(self):
        """Main trading service loop."""
        await self.initialize()

        logger.info("Live trading service started")
        logger.info(f"Starting capital: ${self.capital}")
        logger.info(f"Risk limit: ${RISK_LIMIT}/day")
        logger.info(f"Max leverage: {MAX_LEVERAGE}x")

        while self.is_running:
            try:
                # Scan for opportunities
                await self.scan_opportunities()

                # Manage existing positions
                await self.manage_positions()

                # Check daily loss limit
                if self.daily_pnl <= -RISK_LIMIT:
                    logger.warning(f"Daily loss limit reached (${RISK_LIMIT}). Stopping for today.")
                    self.is_running = False
                    break

                # Wait before next cycle
                await asyncio.sleep(60)  # 1 minute

            except Exception as e:
                logger.error(f"Service error: {e}")
                await asyncio.sleep(60)

    def get_status(self):
        """Get service status."""
        return {
            'service': 'live_trading',
            'running': self.is_running,
            'capital': self.capital,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'positions': len(self.positions),
            'trades_today': len(self.trades_today),
            'last_trade': self.last_trade.isoformat() if self.last_trade else None,
            'open_positions': self.positions,
            'risk_limit': RISK_LIMIT,
            'max_leverage': MAX_LEVERAGE
        }

    async def shutdown(self):
        """Clean shutdown."""
        self.is_running = False
        if self.session:
            await self.session.close()

        # Close all positions at current price (emergency exit)
        for position in self.positions[:]:
            current_price = position['entry_price']  # Use entry price as approximation
            await self.close_position(position, current_price, "emergency_shutdown")

        logger.info("Live trading service shut down")

# Web server for API endpoints
from aiohttp import web

async def health_check(request):
    """Health check endpoint."""
    return web.json_response({
        'status': 'healthy',
        'service': 'live_trading',
        'timestamp': datetime.now().isoformat()
    })

async def status_endpoint(request):
    """Status endpoint."""
    trader = request.app['trader']
    return web.json_response(trader.get_status())

async def trade_endpoint(request):
    """Manual trade trigger."""
    trader = request.app['trader']

    try:
        data = await request.json()
        symbol = data.get('symbol', 'BTCUSDT')

        # Get market data and generate signal
        market_data = await trader.get_market_data(symbol)
        if market_data:
            signal = trader.generate_signal(market_data)
            if signal.get('type'):
                position = trader.calculate_position_size(signal)
                if position:
                    await trader.execute_trade(position)
                    return web.json_response({
                        'status': 'trade_executed',
                        'symbol': symbol,
                        'signal': signal,
                        'position': position
                    })

        return web.json_response({'status': 'no_opportunity_found'})

    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def init_app():
    """Initialize web application."""
    app = web.Application()
    trader = CloudTradingService()
    app['trader'] = trader

    # Routes
    app.router.add_get('/health', health_check)
    app.router.add_get('/status', status_endpoint)
    app.router.add_post('/trade', trade_endpoint)

    # Start background service (commented for safety in demo)
    # asyncio.create_task(trader.run_service())

    return app

if __name__ == "__main__":
    # Run as web service
    app = asyncio.run(init_app())
    web.run_app(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
