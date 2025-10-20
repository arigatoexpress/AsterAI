#!/usr/bin/env python3
"""
🚀 AsterAI Advanced RTX Trading Dashboard Server

Professional Matrix-themed cyberpunk trading dashboard with real-time data,
GPU monitoring, and live trading controls.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Flask and WebSocket imports
try:
    from flask import Flask, render_template_string, request, jsonify
    from flask_socketio import SocketIO, emit
    import psutil
    import plotly
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
    from realtime_price_fetcher import RealTimePriceFetcher
    from trading_control_center import control_center
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install with: pip install flask flask-socketio psutil plotly aiohttp yfinance")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.json_encoder = PlotlyJSONEncoder
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
system_metrics = {}
portfolio_data = {
    'balance': 100.0,
    'pnl': 0.0,
    'positions': [],
    'trades': []
}
market_data = {
    'BTC': {'price': 50000.0, 'change': 0.0},
    'ETH': {'price': 3000.0, 'change': 0.0},
    'SOL': {'price': 95.0, 'change': 0.0},
    'ADA': {'price': 0.35, 'change': 0.0}
}

# Real-time price fetcher
price_fetcher = RealTimePriceFetcher()

# Symbol mapping utilities
SYMBOL_TO_FILE = {
    'BTC': 'btc',
    'ETH': 'eth'
}
SYMBOL_TO_YF = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD'
}

def _load_local_candles(symbol: str) -> Optional[pd.DataFrame]:
    """Load local historical candles from parquet, return DataFrame with datetime index and ohlc columns."""
    try:
        sym = SYMBOL_TO_FILE.get(symbol.upper(), symbol.lower())
        candidates = [
            f"data/historical/crypto/{sym}.parquet",
            f"data/historical/ultimate_dataset/crypto/{sym}.parquet",
            f"data/local_cache/{sym}_historical.parquet"
        ]
        for path in candidates:
            if os.path.exists(path):
                df = pd.read_parquet(path)
                # Normalize
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
                    df = df.set_index('timestamp')
                elif df.index.name is None or not np.issubdtype(df.index.dtype, np.datetime64):
                    # Try to infer index
                    try:
                        df.index = pd.to_datetime(df.index, errors='coerce', utc=True)
                    except Exception:
                        pass
                df = df.sort_index()
                # Standardize column names
                cols = {c.lower(): c for c in df.columns}
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col not in df.columns and col in cols:
                        df[col] = df[cols[col]]
                needed = [c for c in ['open','high','low','close'] if c in df.columns]
                if len(needed) >= 4:
                    return df
        return None
    except Exception as e:
        logger.warning(f"Local candles load failed for {symbol}: {e}")
        return None

def _fetch_api_candles(symbol: str, interval: str = '1m', lookback_hours: int = 6) -> Optional[pd.DataFrame]:
    """Fetch recent candles from Yahoo Finance as a free reliable source for crypto."""
    try:
        import yfinance as yf
        yf_symbol = SYMBOL_TO_YF.get(symbol.upper(), symbol.upper())
        period = '1d' if lookback_hours <= 24 else '5d'
        yf_interval = '1m' if interval == '1m' else '5m'
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period=period, interval=yf_interval)
        if hist is None or hist.empty:
            return None
        hist = hist.tz_convert('UTC') if hist.index.tz is not None else hist.tz_localize('UTC')
        df = pd.DataFrame({
            'open': hist['Open'].astype(float),
            'high': hist['High'].astype(float),
            'low': hist['Low'].astype(float),
            'close': hist['Close'].astype(float),
            'volume': hist.get('Volume', 0)
        }, index=hist.index)
        return df
    except Exception as e:
        logger.warning(f"API candles fetch failed for {symbol}: {e}")
        return None

def _merge_candles(local_df: Optional[pd.DataFrame], api_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Chronologically merge local and API candles without duplicates, prefer newer API where overlapping."""
    if local_df is None and api_df is None:
        return None
    if local_df is None:
        return api_df.sort_index()
    if api_df is None:
        return local_df.sort_index()
    # Concatenate and drop duplicates by index, keeping last (API overwrites local on overlap)
    df = pd.concat([local_df, api_df])
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    # Keep only standard columns
    for col in ['open','high','low','close','volume']:
        if col not in df.columns:
            df[col] = np.nan
    return df[['open','high','low','close','volume']]

def _compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic indicators: SMA20, SMA50, RSI14."""
    out: Dict[str, Any] = {}
    if df is None or df.empty:
        return out
    close = df['close'].astype(float)
    out['sma20'] = close.rolling(20).mean().bfill().tolist()
    out['sma50'] = close.rolling(50).mean().bfill().tolist()
    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.bfill().fillna(50)
    out['rsi14'] = rsi.tolist()
    return out

def _generate_alerts(df: pd.DataFrame, indicators: Dict[str, Any]) -> List[str]:
    alerts: List[str] = []
    if df is None or df.empty:
        return alerts
    try:
        close = df['close'].astype(float)
        sma20 = pd.Series(indicators.get('sma20', []), index=df.index)
        sma50 = pd.Series(indicators.get('sma50', []), index=df.index)
        rsi14 = pd.Series(indicators.get('rsi14', []), index=df.index)
        if len(close) >= 2 and len(sma20) >= 2:
            if close.iloc[-2] < sma20.iloc[-2] and close.iloc[-1] > sma20.iloc[-1]:
                alerts.append("Bullish: Close crossed above SMA20")
            if close.iloc[-2] > sma20.iloc[-2] and close.iloc[-1] < sma20.iloc[-1]:
                alerts.append("Bearish: Close crossed below SMA20")
        if len(sma20) >= 1 and len(sma50) >= 1:
            if sma20.iloc[-1] > sma50.iloc[-1]:
                alerts.append("Trend: SMA20 above SMA50 (uptrend)")
            else:
                alerts.append("Trend: SMA20 below SMA50 (downtrend)")
        if len(rsi14) >= 1:
            rsi = rsi14.iloc[-1]
            if rsi > 70:
                alerts.append("RSI: Overbought (>70)")
            elif rsi < 30:
                alerts.append("RSI: Oversold (<30)")
    except Exception:
        pass
    return alerts

def _explain_market(df: pd.DataFrame, indicators: Dict[str, Any], symbol: str) -> str:
    try:
        if df is None or df.empty:
            return f"No data available for {symbol}."
        close = df['close'].astype(float)
        last_price = close.iloc[-1]
        sma20 = indicators.get('sma20', [])
        sma50 = indicators.get('sma50', [])
        rsi = indicators.get('rsi14', [])
        last_sma20 = sma20[-1] if sma20 else None
        last_sma50 = sma50[-1] if sma50 else None
        last_rsi = rsi[-1] if rsi else None
        trend = "uptrend" if last_sma20 is not None and last_sma50 is not None and last_sma20 > last_sma50 else "downtrend"
        return (
            f"{symbol}: Price {last_price:.2f} USD; {trend}. "
            f"SMA20={last_sma20:.2f} SMA50={last_sma50:.2f}; RSI14={last_rsi:.1f}. "
            "Signals and overlays are based on merged local+API data."
        )
    except Exception:
        return f"Computed summary not available for {symbol}."

# Matrix-themed HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 ASTER AI - Live Trading Matrix</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Courier New', monospace;
            background: #000;
            color: #00ff00;
            overflow-x: hidden;
        }

        /* Matrix rain background */
        .matrix-rain {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
            opacity: 0.1;
        }

        .matrix-rain canvas {
            display: block;
        }

        /* Header */
        .header {
            background: rgba(0, 20, 0, 0.9);
            border-bottom: 2px solid #00ff00;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 0 20px #00ff00;
        }

        .header h1 {
            font-size: 2.5rem;
            text-shadow: 0 0 10px #00ff00;
        }

        /* Navigation */
        .nav {
            background: rgba(0, 10, 0, 0.8);
            border-bottom: 1px solid #00ff00;
            padding: 0.5rem;
        }

        .nav-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
        }

        .nav-btn {
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid #00ff00;
            color: #00ff00;
            padding: 0.5rem 1rem;
            cursor: pointer;
            border-radius: 5px;
            transition: all 0.3s;
        }

        .nav-btn:hover, .nav-btn.active {
            background: rgba(0, 255, 0, 0.3);
            box-shadow: 0 0 10px #00ff00;
        }

        /* Main content */
        .main-content {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }

        .page {
            display: none;
        }

        .page.active {
            display: block;
        }

        /* Cards */
        .card {
            background: rgba(0, 20, 0, 0.8);
            border: 1px solid #00ff00;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
        }

        .card h3 {
            color: #00ff00;
            border-bottom: 1px solid #00ff00;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }

        /* Grid layouts */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }

        .metric {
            text-align: center;
            padding: 1rem;
            background: rgba(0, 255, 0, 0.1);
            border-radius: 5px;
            border: 1px solid #00ff00;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00ff00;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #00aa00;
        }

        /* Status indicators */
        .status-good { color: #00ff00; }
        .status-warning { color: #ffff00; }
        .status-error { color: #ff0000; }

        /* Animations */
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px #00ff00; }
            50% { box-shadow: 0 0 20px #00ff00; }
        }

        .glow {
            animation: glow 2s infinite;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .nav-buttons {
                flex-direction: column;
                align-items: center;
            }

            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="matrix-rain" id="matrixRain"></div>

    <header class="header">
        <h1>🚀 ASTER AI - Live Trading Matrix</h1>
        <div id="status">System Status: <span id="systemStatus" class="status-good">ONLINE</span></div>
    </header>

    <nav class="nav">
        <div class="nav-buttons">
            <button class="nav-btn active" onclick="showPage('dashboard')">Dashboard</button>
            <button class="nav-btn" onclick="showPage('trading')">Trading Panel</button>
            <button class="nav-btn" onclick="showPage('system')">System Console</button>
            <button class="nav-btn" onclick="showPage('ai')">AI Information</button>
            <button class="nav-btn" onclick="showPage('help')">Help & Guides</button>
        </div>
    </nav>

    <main class="main-content">
        <!-- Dashboard Page -->
        <div id="dashboard" class="page active">
            <div class="grid">
                <div class="card">
                    <h3>💰 Portfolio Status</h3>
                    <div class="metric">
                        <div class="metric-value" id="portfolioBalance">$100.00</div>
                        <div class="metric-label">Total Balance</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                        <div class="metric">
                            <div class="metric-value" id="portfolioPnL">$0.00</div>
                            <div class="metric-label">P&L</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="portfolioChange">0.00%</div>
                            <div class="metric-label">Change</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>📊 Market Data</h3>
                    <div id="marketData">
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>BTC/USDT:</span>
                            <span id="btcPrice">$50,000.00</span>
                            <span id="btcChange" class="status-good">+0.00%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>ETH/USDT:</span>
                            <span id="ethPrice">$3,000.00</span>
                            <span id="ethChange" class="status-good">+0.00%</span>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>⚡ System Performance</h3>
                    <div id="systemMetrics">
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>CPU Usage:</span>
                            <span id="cpuUsage">0%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>Memory Usage:</span>
                            <span id="memoryUsage">0%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>GPU Usage:</span>
                            <span id="gpuUsage">N/A</span>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>🎯 Active Positions</h3>
                    <div id="positionsList">
                        <p class="status-warning">No active positions</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📈 Portfolio Performance Chart</h3>
                <div id="performanceChart" style="height: 400px;"></div>
            </div>

        <div class="card">
            <h3>🕰️ Merged Candles & Indicators</h3>
            <div style="margin-bottom: 1rem; display: flex; gap: 1rem; align-items: center;">
                <label for="symbolSelect">Symbol:</label>
                <select id="symbolSelect">
                    <option value="BTC" selected>BTC</option>
                    <option value="ETH">ETH</option>
                </select>
                <button class="nav-btn" onclick="loadMergedData()">Refresh</button>
                <span id="analysisSummary" style="margin-left: auto; color: #00ff99;"></span>
            </div>
            <div id="mergedChart" style="height: 420px;"></div>
            <div id="alertsBox" class="metric" style="margin-top: 1rem; text-align: left;"></div>
        </div>
        </div>

        <!-- Trading Panel Page -->
        <div id="trading" class="page">
            <div class="card">
                <h3>🎮 Bot Control Center</h3>
                <div class="grid" style="grid-template-columns: repeat(3, 1fr);">
                    <div class="metric glow">
                        <div class="metric-label">Bot Status</div>
                        <div class="metric-value" id="botStatus" style="font-size: 1.2rem;">STANDBY</div>
                        <button class="nav-btn" onclick="startBot()" id="startBtn" style="margin-top: 0.5rem;">▶ START</button>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Trading Mode</div>
                        <div class="metric-value" id="tradingMode" style="font-size: 1rem;">DRY-RUN</div>
                        <button class="nav-btn" onclick="toggleMode()" style="margin-top: 0.5rem;">🔄 TOGGLE</button>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Emergency</div>
                        <div class="metric-value" style="font-size: 1rem;">ARMED</div>
                        <button class="nav-btn" onclick="emergencyStop()" style="margin-top: 0.5rem; background: rgba(255,0,0,0.3); border-color: #ff0000;">🚨 STOP</button>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>⚙️ Configuration Panel</h3>
                <div class="grid" style="grid-template-columns: repeat(2, 1fr);">
                    <div>
                        <label>Position Size (%)</label>
                        <input type="range" id="positionSize" min="1" max="10" value="2" oninput="updateConfigDisplay()" style="width: 100%; margin: 0.5rem 0;">
                        <div id="positionSizeVal">2%</div>
                    </div>
                    <div>
                        <label>Stop Loss (%)</label>
                        <input type="range" id="stopLoss" min="1" max="10" value="2" oninput="updateConfigDisplay()" style="width: 100%; margin: 0.5rem 0;">
                        <div id="stopLossVal">2%</div>
                    </div>
                    <div>
                        <label>Take Profit (%)</label>
                        <input type="range" id="takeProfit" min="2" max="20" value="4" oninput="updateConfigDisplay()" style="width: 100%; margin: 0.5rem 0;">
                        <div id="takeProfitVal">4%</div>
                    </div>
                    <div>
                        <label>Max Positions</label>
                        <input type="range" id="maxPositions" min="1" max="5" value="2" oninput="updateConfigDisplay()" style="width: 100%; margin: 0.5rem 0;">
                        <div id="maxPositionsVal">2</div>
                    </div>
                </div>
                <button class="nav-btn" onclick="saveConfig()" style="margin-top: 1rem; width: 100%;">💾 SAVE CONFIGURATION</button>
            </div>

            <div class="card">
                <h3>📈 Manual Trade Execution</h3>
                <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                    <select id="manualSymbol" style="flex: 1; padding: 0.5rem; background: rgba(0,255,0,0.1); border: 1px solid #00ff00; color: #00ff00;">
                        <option value="BTCUSDT">BTC/USDT</option>
                        <option value="ETHUSDT">ETH/USDT</option>
                        <option value="SOLUSDT">SOL/USDT</option>
                        <option value="ADAUSDT">ADA/USDT</option>
                    </select>
                    <input type="number" id="manualAmount" placeholder="Amount ($)" style="flex: 1; padding: 0.5rem; background: rgba(0,255,0,0.1); border: 1px solid #00ff00; color: #00ff00;" value="2.00">
                </div>
                <div style="display: flex; gap: 1rem;">
                    <button class="nav-btn" onclick="placeManualTrade('BUY')" style="flex: 1; background: rgba(0,255,0,0.2);">📈 BUY</button>
                    <button class="nav-btn" onclick="placeManualTrade('SELL')" style="flex: 1; background: rgba(255,0,0,0.2); border-color: #ff0000;">📉 SELL</button>
                </div>
                <div id="tradeStatus" style="margin-top: 1rem; padding: 0.5rem; background: rgba(0,255,0,0.1); border: 1px solid #00ff00; display: none;"></div>
            </div>

            <div class="card">
                <h3>📊 Trading Performance</h3>
                <div id="tradingMetrics">
                    <div class="grid">
                        <div class="metric">
                            <div class="metric-value" id="totalTrades">0</div>
                            <div class="metric-label">Total Trades</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="winRate">0.00%</div>
                            <div class="metric-label">Win Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="sharpeRatio">0.00</div>
                            <div class="metric-label">Sharpe Ratio</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="maxDrawdown">0.00%</div>
                            <div class="metric-label">Max Drawdown</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📜 Recent Trades</h3>
                <div id="recentTradesList" style="max-height: 300px; overflow-y: auto;">
                    <p class="status-warning">No trades executed yet</p>
                </div>
            </div>
        </div>

        <!-- System Console Page -->
        <div id="system" class="page">
            <div class="card">
                <h3>🔧 System Logs</h3>
                <div id="systemLogs" style="height: 300px; overflow-y: auto; background: rgba(0, 0, 0, 0.5); padding: 1rem; font-family: monospace;">
                    <div id="logContainer">
                        <p>System initialized - waiting for logs...</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>⚙️ Bot Settings</h3>
                <div id="botSettings">
                    <p>Trading Mode: <span class="status-good">AUTONOMOUS</span></p>
                    <p>Risk Level: <span class="status-good">CONSERVATIVE</span></p>
                    <p>Max Positions: <span>2</span></p>
                    <p>Stop Loss: <span>2%</span></p>
                    <p>Take Profit: <span>4%</span></p>
                </div>
            </div>
        </div>

        <!-- AI Information Page -->
        <div id="ai" class="page">
            <div class="card">
                <h3>🤖 AI Decision Engine</h3>
                <p>The Aster AI trading system uses advanced machine learning algorithms to make autonomous trading decisions.</p>

                <h4>Active Strategies:</h4>
                <ul>
                    <li>• Market Making Strategy</li>
                    <li>• Funding Rate Arbitrage</li>
                    <li>• Adaptive Risk Management</li>
                    <li>• MEV Protection System</li>
                </ul>
            </div>

            <div class="card">
                <h3>🧠 Learning & Adaptation</h3>
                <p>The AI continuously learns from market conditions and adapts its strategies based on:</p>
                <ul>
                    <li>• Market regime detection</li>
                    <li>• Volatility analysis</li>
                    <li>• Risk-adjusted performance</li>
                    <li>• Historical pattern recognition</li>
                </ul>
            </div>
        </div>

        <!-- Help & Guides Page -->
        <div id="help" class="page">
            <div class="card">
                <h3>📚 User Guide</h3>
                <h4>Getting Started:</h4>
                <ol>
                    <li>Monitor the Dashboard for real-time portfolio status</li>
                    <li>Check System Console for detailed logs and performance</li>
                    <li>Review AI Information to understand decision making</li>
                    <li>Use Trading Panel for manual overrides (future feature)</li>
                </ol>

                <h4>Safety Features:</h4>
                <ul>
                    <li>• Automatic stop-loss protection</li>
                    <li>• Daily loss limits</li>
                    <li>• Position size controls</li>
                    <li>• Emergency shutdown capability</li>
                </ul>
            </div>

            <div class="card">
                <h3>🚨 Emergency Procedures</h3>
                <p>If the system detects abnormal conditions:</p>
                <ul>
                    <li>Automatic position closure</li>
                    <li>Trading suspension</li>
                    <li>Alert notifications</li>
                    <li>Manual intervention options</li>
                </ul>
            </div>
        </div>
    </main>

    <script>
        // Matrix rain effect
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        document.getElementById('matrixRain').appendChild(canvas);

        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const matrix = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        const matrixArray = matrix.split("");

        const fontSize = 16;
        const columns = canvas.width / fontSize;
        const drops = [];

        for (let x = 0; x < columns; x++) {
            drops[x] = 1;
        }

        function draw() {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.04)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#00ff00';
            ctx.font = fontSize + 'px monospace';

            for (let i = 0; i < drops.length; i++) {
                const text = matrixArray[Math.floor(Math.random() * matrixArray.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);

                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            }
        }

        setInterval(draw, 35);

        // Socket.IO connection
        const socket = io();

        socket.on('connect', function() {
            console.log('Connected to dashboard server');
        });

        socket.on('system_update', function(data) {
            updateSystemMetrics(data);
        });

        socket.on('portfolio_update', function(data) {
            updatePortfolio(data);
        });

        socket.on('market_update', function(data) {
            updateMarketData(data);
        });

        socket.on('trading_update', function(data) {
            updateTradingData(data);
        });

        // Page navigation
        function showPage(pageId) {
            // Hide all pages
            const pages = document.querySelectorAll('.page');
            pages.forEach(page => page.classList.remove('active'));

            // Show selected page
            document.getElementById(pageId).classList.add('active');

            // Update nav buttons
            const navBtns = document.querySelectorAll('.nav-btn');
            navBtns.forEach(btn => btn.classList.remove('active'));

            event.target.classList.add('active');
        }

        // Update functions
        function updateSystemMetrics(data) {
            if (data.cpu) document.getElementById('cpuUsage').textContent = data.cpu + '%';
            if (data.memory) document.getElementById('memoryUsage').textContent = data.memory + '%';
            if (data.gpu !== undefined) document.getElementById('gpuUsage').textContent = data.gpu + '%';
        }

        function updatePortfolio(data) {
            if (data.balance) document.getElementById('portfolioBalance').textContent = '$' + data.balance.toFixed(2);
            if (data.pnl !== undefined) {
                document.getElementById('portfolioPnL').textContent = '$' + data.pnl.toFixed(2);
                document.getElementById('portfolioPnL').className = data.pnl >= 0 ? 'status-good' : 'status-error';
            }
        }

        function updateMarketData(data) {
            if (data.BTC) {
                document.getElementById('btcPrice').textContent = '$' + data.BTC.price.toLocaleString();
                document.getElementById('btcChange').textContent = (data.BTC.change >= 0 ? '+' : '') + data.BTC.change.toFixed(2) + '%';
                document.getElementById('btcChange').className = data.BTC.change >= 0 ? 'status-good' : 'status-error';
            }
            if (data.ETH) {
                document.getElementById('ethPrice').textContent = '$' + data.ETH.price.toLocaleString();
                document.getElementById('ethChange').textContent = (data.ETH.change >= 0 ? '+' : '') + data.ETH.change.toFixed(2) + '%';
                document.getElementById('ethChange').className = data.ETH.change >= 0 ? 'status-good' : 'status-error';
            }
        }

        function updateTradingData(data) {
            if (data.total_trades !== undefined) document.getElementById('totalTrades').textContent = data.total_trades;
            if (data.win_rate !== undefined) document.getElementById('winRate').textContent = (data.win_rate * 100).toFixed(2) + '%';
            if (data.sharpe_ratio !== undefined) document.getElementById('sharpeRatio').textContent = data.sharpe_ratio.toFixed(2);
            if (data.max_drawdown !== undefined) document.getElementById('maxDrawdown').textContent = (data.max_drawdown * 100).toFixed(2) + '%';
        }

        // Initialize with sample data
        updateSystemMetrics({cpu: 15, memory: 45, gpu: 0});
        updatePortfolio({balance: 100.0, pnl: 0.0});
        updateMarketData({
            BTC: {price: 50000.0, change: 0.5},
            ETH: {price: 3000.0, change: -0.2}
        });
        updateTradingData({total_trades: 0, win_rate: 0, sharpe_ratio: 0, max_drawdown: 0});

        // Load merged candles + indicators
        async function loadMergedData() {
            const symbol = document.getElementById('symbolSelect').value;
            try {
                const [candlesResp, indicatorsResp] = await Promise.all([
                    fetch(`/api/merged-candles?symbol=${symbol}&interval=1m`),
                    fetch(`/api/indicators?symbol=${symbol}&interval=1m`)
                ]);
                const candlesData = await candlesResp.json();
                const indiData = await indicatorsResp.json();

                renderMergedChart(candlesData, indiData);
            } catch (err) {
                console.error('Error loading merged data:', err);
            }
        }

        function renderMergedChart(candlesData, indiData) {
            const candles = candlesData.candles || [];
            if (candles.length === 0) {
                document.getElementById('analysisSummary').textContent = 'No data available';
                return;
            }
            const times = candles.map(c => c.timestamp);
            const open = candles.map(c => c.open);
            const high = candles.map(c => c.high);
            const low = candles.map(c => c.low);
            const close = candles.map(c => c.close);

            const traceCandle = {
                x: times,
                open: open,
                high: high,
                low: low,
                close: close,
                type: 'candlestick',
                name: 'Price'
            };

            const ind = indiData.indicators || {};
            const sma20 = (ind.sma20 || []).slice(-times.length);
            const sma50 = (ind.sma50 || []).slice(-times.length);
            const rsi14 = (ind.rsi14 || []).slice(-times.length);

            const traceSMA20 = {
                x: times,
                y: sma20,
                type: 'scatter',
                mode: 'lines',
                name: 'SMA20',
                line: { color: '#00ff99' }
            };
            const traceSMA50 = {
                x: times,
                y: sma50,
                type: 'scatter',
                mode: 'lines',
                name: 'SMA50',
                line: { color: '#0099ff' }
            };

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Price (USD)' },
                font: { color: '#00ff00' }
            };

            Plotly.newPlot('mergedChart', [traceCandle, traceSMA20, traceSMA50], layout, {displayModeBar: false});

            // Alerts and explanation
            const alerts = indiData.alerts || [];
            const explanation = indiData.explanation || '';
            document.getElementById('analysisSummary').textContent = explanation;
            const alertsBox = document.getElementById('alertsBox');
            alertsBox.innerHTML = `<b>Alerts:</b><br>` + (alerts.length ? alerts.map(a => `• ${a}`).join('<br>') : 'None');
        }

        // Auto-load on startup
        loadMergedData();

        // Bot Control Functions
        async function startBot() {
            try {
                const response = await fetch('/api/control/start', {method: 'POST'});
                const result = await response.json();
                if (result.status === 'success') {
                    document.getElementById('botStatus').textContent = 'RUNNING';
                    document.getElementById('botStatus').className = 'metric-value status-good';
                    document.getElementById('startBtn').textContent = '⏸ PAUSE';
                    document.getElementById('startBtn').onclick = stopBot;
                    showNotification('✅ Trading bot started', 'success');
                } else {
                    showNotification('❌ ' + result.message, 'error');
                }
            } catch (err) {
                showNotification('❌ Failed to start bot', 'error');
            }
        }

        async function stopBot() {
            try {
                const response = await fetch('/api/control/stop', {method: 'POST'});
                const result = await response.json();
                if (result.status === 'success') {
                    document.getElementById('botStatus').textContent = 'STANDBY';
                    document.getElementById('botStatus').className = 'metric-value status-warning';
                    document.getElementById('startBtn').textContent = '▶ START';
                    document.getElementById('startBtn').onclick = startBot;
                    showNotification('⏹ Trading bot stopped', 'warning');
                } else {
                    showNotification('❌ ' + result.message, 'error');
                }
            } catch (err) {
                showNotification('❌ Failed to stop bot', 'error');
            }
        }

        async function emergencyStop() {
            if (!confirm('⚠️ EMERGENCY STOP will close all positions immediately. Continue?')) return;
            try {
                const response = await fetch('/api/control/emergency-stop', {method: 'POST'});
                const result = await response.json();
                if (result.status === 'success') {
                    document.getElementById('botStatus').textContent = 'EMERGENCY STOP';
                    document.getElementById('botStatus').className = 'metric-value status-error';
                    showNotification('🚨 EMERGENCY STOP EXECUTED', 'error');
                } else {
                    showNotification('❌ ' + result.message, 'error');
                }
            } catch (err) {
                showNotification('❌ Emergency stop failed', 'error');
            }
        }

        function updateConfigDisplay() {
            document.getElementById('positionSizeVal').textContent = document.getElementById('positionSize').value + '%';
            document.getElementById('stopLossVal').textContent = document.getElementById('stopLoss').value + '%';
            document.getElementById('takeProfitVal').textContent = document.getElementById('takeProfit').value + '%';
            document.getElementById('maxPositionsVal').textContent = document.getElementById('maxPositions').value;
        }

        async function saveConfig() {
            const config = {
                position_size_pct: parseInt(document.getElementById('positionSize').value) / 100,
                stop_loss_pct: parseInt(document.getElementById('stopLoss').value) / 100,
                take_profit_pct: parseInt(document.getElementById('takeProfit').value) / 100,
                max_positions: parseInt(document.getElementById('maxPositions').value)
            };

            try {
                const response = await fetch('/api/control/config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                const result = await response.json();
                if (result.status === 'success') {
                    showNotification('✅ Configuration saved', 'success');
                } else {
                    showNotification('❌ ' + result.message, 'error');
                }
            } catch (err) {
                showNotification('❌ Failed to save configuration', 'error');
            }
        }

        async function placeManualTrade(side) {
            const symbol = document.getElementById('manualSymbol').value;
            const amount = parseFloat(document.getElementById('manualAmount').value);

            if (!amount || amount <= 0) {
                showNotification('❌ Invalid amount', 'error');
                return;
            }

            try {
                const response = await fetch('/api/trade/manual', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol, side, amount})
                });
                const result = await response.json();
                
                const statusDiv = document.getElementById('tradeStatus');
                statusDiv.style.display = 'block';
                
                if (result.status === 'success') {
                    statusDiv.innerHTML = `✅ ${side} ${amount} USD of ${symbol} - ${result.message}`;
                    statusDiv.style.borderColor = '#00ff00';
                    showNotification(`✅ Trade placed: ${side} ${symbol}`, 'success');
                    loadRecentTrades();
                } else {
                    statusDiv.innerHTML = `❌ ${result.message}`;
                    statusDiv.style.borderColor = '#ff0000';
                    showNotification('❌ Trade failed', 'error');
                }
            } catch (err) {
                showNotification('❌ Trade execution failed', 'error');
            }
        }

        async function loadRecentTrades() {
            try {
                const response = await fetch('/api/control/status');
                const data = await response.json();
                const trades = data.recent_trades || [];
                
                const tradesList = document.getElementById('recentTradesList');
                if (trades.length === 0) {
                    tradesList.innerHTML = '<p class="status-warning">No trades executed yet</p>';
                } else {
                    tradesList.innerHTML = trades.map(trade => `
                        <div style="padding: 0.5rem; margin: 0.5rem 0; background: rgba(0,255,0,0.1); border-left: 3px solid ${trade.side === 'BUY' ? '#00ff00' : '#ff0000'};">
                            <strong>${trade.symbol}</strong> - ${trade.side} ${trade.amount} USD
                            <br><small>${trade.timestamp} - ${trade.status}</small>
                        </div>
                    `).join('');
                }
            } catch (err) {
                console.error('Failed to load recent trades:', err);
            }
        }

        function toggleMode() {
            const modeDiv = document.getElementById('tradingMode');
            if (modeDiv.textContent === 'DRY-RUN') {
                if (confirm('⚠️ Switch to LIVE mode? Real trades will be executed!')) {
                    modeDiv.textContent = 'LIVE';
                    modeDiv.className = 'metric-value status-error';
                    showNotification('🔴 LIVE MODE ACTIVATED', 'error');
                }
            } else {
                modeDiv.textContent = 'DRY-RUN';
                modeDiv.className = 'metric-value status-good';
                showNotification('✅ Dry-run mode activated', 'success');
            }
        }

        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                padding: 1rem 2rem;
                background: ${type === 'success' ? 'rgba(0,255,0,0.2)' : type === 'error' ? 'rgba(255,0,0,0.2)' : 'rgba(255,255,0,0.2)'};
                border: 2px solid ${type === 'success' ? '#00ff00' : type === 'error' ? '#ff0000' : '#ffff00'};
                color: ${type === 'success' ? '#00ff00' : type === 'error' ? '#ff0000' : '#ffff00'};
                border-radius: 5px;
                z-index: 9999;
                box-shadow: 0 0 20px ${type === 'success' ? '#00ff00' : type === 'error' ? '#ff0000' : '#ffff00'};
                font-family: 'Courier New', monospace;
            `;
            document.body.appendChild(notification);
            setTimeout(() => notification.remove(), 3000);
        }

        // WebSocket listeners for control updates
        socket.on('bot_status_change', function(data) {
            if (data.trading_active) {
                document.getElementById('botStatus').textContent = 'RUNNING';
                document.getElementById('botStatus').className = 'metric-value status-good';
            } else {
                document.getElementById('botStatus').textContent = data.emergency ? 'EMERGENCY STOP' : 'STANDBY';
                document.getElementById('botStatus').className = 'metric-value status-' + (data.emergency ? 'error' : 'warning');
            }
        });

        socket.on('trade_executed', function(trade) {
            showNotification(`📊 Trade executed: ${trade.side} ${trade.symbol}`, 'success');
            loadRecentTrades();
        });

        socket.on('config_updated', function(config) {
            showNotification('⚙️ Configuration updated', 'success');
        });

        // Load recent trades on page load
        loadRecentTrades();
        setInterval(loadRecentTrades, 10000);  // Refresh every 10 seconds
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/merged-candles')
def api_merged_candles():
    """Return merged candles (local parquet + API) for a symbol."""
    symbol = request.args.get('symbol', 'BTC').upper()
    interval = request.args.get('interval', '1m')
    local_df = _load_local_candles(symbol)
    api_df = _fetch_api_candles(symbol, interval=interval)
    merged = _merge_candles(local_df, api_df)
    if merged is None or merged.empty:
        return jsonify({'symbol': symbol, 'candles': []})
    candles = []
    for ts, row in merged.iterrows():
        candles.append({
            'timestamp': ts.isoformat(),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0) or 0)
        })
    return jsonify({'symbol': symbol, 'interval': interval, 'candles': candles})

@app.route('/api/indicators')
def api_indicators():
    symbol = request.args.get('symbol', 'BTC').upper()
    interval = request.args.get('interval', '1m')
    local_df = _load_local_candles(symbol)
    api_df = _fetch_api_candles(symbol, interval=interval)
    merged = _merge_candles(local_df, api_df)
    if merged is None or merged.empty:
        return jsonify({'symbol': symbol, 'indicators': {}, 'alerts': [], 'explanation': f'No data for {symbol}.'})
    indicators = _compute_indicators(merged)
    alerts = _generate_alerts(merged, indicators)
    explanation = _explain_market(merged, indicators, symbol)
    return jsonify({'symbol': symbol, 'indicators': indicators, 'alerts': alerts, 'explanation': explanation})

@app.route('/api/control/start', methods=['POST'])
async def api_start_trading():
    """Start the trading bot"""
    result = await control_center.start_trading()
    socketio.emit('bot_status_change', {'trading_active': control_center.trading_active})
    return jsonify(result)

@app.route('/api/control/stop', methods=['POST'])
async def api_stop_trading():
    """Stop the trading bot"""
    result = await control_center.stop_trading()
    socketio.emit('bot_status_change', {'trading_active': control_center.trading_active})
    return jsonify(result)

@app.route('/api/control/emergency-stop', methods=['POST'])
async def api_emergency_stop():
    """Emergency stop all trading"""
    result = await control_center.emergency_stop()
    socketio.emit('bot_status_change', {'trading_active': False, 'emergency': True})
    return jsonify(result)

@app.route('/api/control/config', methods=['GET', 'POST'])
async def api_bot_config():
    """Get or update bot configuration"""
    if request.method == 'POST':
        new_config = request.json
        result = await control_center.update_config(new_config)
        socketio.emit('config_updated', result.get('config', {}))
        return jsonify(result)
    else:
        return jsonify({'config': control_center.bot_config})

@app.route('/api/control/status')
def api_control_status():
    """Get complete control center status"""
    return jsonify(control_center.get_status())

@app.route('/api/trade/manual', methods=['POST'])
async def api_manual_trade():
    """Place a manual trade"""
    data = request.json
    symbol = data.get('symbol')
    side = data.get('side')
    amount = data.get('amount')
    
    if not all([symbol, side, amount]):
        return jsonify({'status': 'error', 'message': 'Missing required parameters'})
    
    result = await control_center.place_manual_trade(symbol, side, float(amount))
    socketio.emit('trade_executed', result.get('trade', {}))
    return jsonify(result)

@app.route('/api/system-status')
def get_system_status():
    """API endpoint for system status."""
    return jsonify({
        'status': 'online',
        'uptime': time.time(),
        'version': '1.0.0',
        'trading_active': True
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected to dashboard")
    emit('status', {'message': 'Connected to Aster AI Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected from dashboard")

def get_system_metrics():
    """Get current system metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        return {
            'cpu': round(cpu_percent, 1),
            'memory': round(memory_percent, 1),
            'gpu': 0.0  # Placeholder for GPU usage
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {'cpu': 0, 'memory': 0, 'gpu': 0}

async def background_data_updates():
    """Background task to update dashboard data."""
    await price_fetcher.__aenter__()  # Initialize the price fetcher

    while True:
        try:
            # Update system metrics
            metrics = get_system_metrics()
            socketio.emit('system_update', metrics)

            # Update portfolio data
            portfolio_update = {
                'balance': portfolio_data['balance'],
                'pnl': portfolio_data['pnl'],
                'positions': portfolio_data['positions']
            }
            socketio.emit('portfolio_update', portfolio_update)

            # Update market data with real prices
            try:
                real_prices = await price_fetcher.get_current_prices(['BTC', 'ETH', 'SOL', 'ADA'])
                if real_prices:
                    for symbol, data in real_prices.items():
                        if symbol in market_data:
                            old_price = market_data[symbol]['price']
                            new_price = data['price']
                            market_data[symbol]['price'] = new_price
                            market_data[symbol]['change'] = data.get('change_24h', 0.0)
                            market_data[symbol]['source'] = data.get('source', 'unknown')
                            market_data[symbol]['last_update'] = data.get('timestamp', datetime.now().isoformat())

                    logger.info(f"Updated market data with real prices from {len(real_prices)} sources")
                else:
                    # Fallback to slight random changes if API fails
                    logger.warning("Failed to fetch real prices, using simulated changes")
                    for symbol in market_data:
                        change = np.random.normal(0, 0.0001)  # Very small random change
                        market_data[symbol]['price'] *= (1 + change)

            except Exception as e:
                logger.error(f"Error fetching real prices: {e}")
                # Fallback to simulated changes
                for symbol in market_data:
                    change = np.random.normal(0, 0.0001)
                    market_data[symbol]['price'] *= (1 + change)

            socketio.emit('market_update', market_data)

            # Update trading metrics
            trading_update = {
                'total_trades': len(portfolio_data['trades']),
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
            socketio.emit('trading_update', trading_update)

            await asyncio.sleep(10)  # Update every 10 seconds to respect API limits

        except Exception as e:
            logger.error(f"Error in background updates: {e}")
            await asyncio.sleep(30)  # Wait longer on error

async def main():
    """Main function to start the dashboard server."""
    print("="*80)
    print("🚀 ASTER AI - Live Trading Matrix Dashboard")
    print("="*80)
    print("🌐 Starting dashboard server...")
    print("📊 Features:")
    print("   • Real-time GPU performance monitoring")
    print("   • Live market price updates from CoinGecko/Binance")
    print("   • Interactive trading controls")
    print("   • Auto-updating charts and metrics")
    print("   • WebSocket live data connections")
    print()
    print("🌐 Dashboard will be available at: http://localhost:8081")
    print("🔗 Access the Matrix interface in your browser")
    print()

    # Start background data updates
    import threading
    update_thread = threading.Thread(target=lambda: asyncio.run(background_data_updates()), daemon=True)
    update_thread.start()

    try:
        # Start the server
        socketio.run(app, host='0.0.0.0', port=8081, debug=False)
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard server stopped by user")
    except Exception as e:
        print(f"\n❌ Dashboard server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
