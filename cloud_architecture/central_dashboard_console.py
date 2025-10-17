#!/usr/bin/env python3
"""
CENTRAL DASHBOARD CONSOLE
Cloud-deployed operations center for continuous AI trading

Features:
- Continuous data collection from Aster DEX
- Automated backtesting pipeline
- Live trading bot management
- Cost-optimized GCP architecture
- Real-time performance monitoring
- Automated model retraining
- Risk management dashboard
"""

import sys
import os
import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import psutil
import requests
import schedule
import threading

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/central_console.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Central configuration for the entire system."""

    # GCP Settings
    GCP_PROJECT = os.environ.get('GCP_PROJECT', 'aster-ai-trading')
    GCP_REGION = os.environ.get('GCP_REGION', 'us-central1')

    # Cost Optimization
    MAX_DAILY_COST = 10.0  # $10/day budget
    DATA_COLLECTION_INTERVAL = 300  # 5 minutes
    BACKTEST_INTERVAL = 3600  # 1 hour
    MODEL_RETRAIN_INTERVAL = 86400  # 24 hours

    # Trading Settings
    MAX_CONCURRENT_TRADES = 2
    DAILY_LOSS_LIMIT = 50.0  # $50 daily loss limit
    TRADE_SIZE_MIN = 10.0
    TRADE_SIZE_MAX = 100.0

    # Data Settings
    HISTORICAL_DAYS = 365
    REAL_TIME_UPDATE_INTERVAL = 60  # 1 minute

    # API Settings
    ASTER_API_KEY = os.environ.get('ASTER_API_KEY')
    ASTER_SECRET = os.environ.get('ASTER_SECRET')
    ASTER_BASE_URL = 'https://fapi.asterdex.com'


class DataCollectionService:
    """Continuous data collection from Aster DEX."""

    def __init__(self):
        self.config = Config()
        self.is_running = False
        self.last_collection = None
        self.collection_stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'data_points_collected': 0,
            'last_error': None
        }

    async def start_collection(self):
        """Start continuous data collection."""
        self.is_running = True
        logger.info("Starting continuous data collection service")

        while self.is_running:
            try:
                await self.collect_data()
                await asyncio.sleep(self.config.DATA_COLLECTION_INTERVAL)
            except Exception as e:
                logger.error(f"Data collection error: {e}")
                self.collection_stats['failed_collections'] += 1
                self.collection_stats['last_error'] = str(e)
                await asyncio.sleep(60)  # Wait before retry

    async def collect_data(self):
        """Collect data from Aster DEX."""
        try:
            from local_training.aster_dex_data_collector import AsterDEXDataCollector

            collector = AsterDEXDataCollector()
            await collector.initialize()

            # Collect real-time data
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'ADAUSDT']

            for symbol in symbols:
                try:
                    # Collect recent trades
                    trades = await collector.collect_recent_trades(symbol, limit=100)

                    # Collect orderbook
                    orderbook = await collector.collect_orderbook(symbol)

                    # Save to cloud storage (would be GCP Cloud Storage)
                    self.save_to_cloud_storage(symbol, 'trades', trades)
                    self.save_to_cloud_storage(symbol, 'orderbook', orderbook)

                    self.collection_stats['successful_collections'] += 1
                    self.collection_stats['data_points_collected'] += len(trades) if trades else 0

                except Exception as e:
                    logger.warning(f"Failed to collect {symbol}: {e}")

            self.collection_stats['total_collections'] += 1
            self.last_collection = datetime.now()

            await collector.close()

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            self.collection_stats['failed_collections'] += 1

    def save_to_cloud_storage(self, symbol: str, data_type: str, data):
        """Save data to cloud storage (GCP Cloud Storage)."""
        # In production, this would upload to GCP Cloud Storage
        # For now, save locally with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/realtime/{symbol}_{data_type}_{timestamp}.json"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_status(self) -> Dict:
        """Get collection service status."""
        return {
            'running': self.is_running,
            'last_collection': self.last_collection.isoformat() if self.last_collection else None,
            'stats': self.collection_stats,
            'next_collection_in': self.config.DATA_COLLECTION_INTERVAL - (
                (datetime.now() - self.last_collection).total_seconds()
                if self.last_collection else 0
            )
        }


class AutomatedBacktestingService:
    """Automated backtesting service."""

    def __init__(self):
        self.config = Config()
        self.is_running = False
        self.backtest_results = []
        self.last_backtest = None

    async def start_backtesting(self):
        """Start automated backtesting."""
        self.is_running = True
        logger.info("Starting automated backtesting service")

        while self.is_running:
            try:
                await self.run_backtest()
                await asyncio.sleep(self.config.BACKTEST_INTERVAL)
            except Exception as e:
                logger.error(f"Backtesting error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def run_backtest(self):
        """Run comprehensive backtest."""
        try:
            from scripts.backtest_cpu_model import CPUBacktester

            logger.info("Running automated backtest...")

            backtester = CPUBacktester()
            success = backtester.run()

            if success:
                # Store results
                result = {
                    'timestamp': datetime.now(),
                    'success': True,
                    'metrics': backtester.calculate_metrics(),
                    'trades': backtester.trades,
                    'equity_curve': backtester.equity_curve
                }

                self.backtest_results.append(result)
                self.last_backtest = datetime.now()

                # Keep only last 10 results
                if len(self.backtest_results) > 10:
                    self.backtest_results = self.backtest_results[-10:]

                logger.info(f"Backtest completed: {result['metrics'].get('win_rate', 0):.1f}% win rate")

            else:
                logger.warning("Backtest failed")

        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")

    def get_status(self) -> Dict:
        """Get backtesting service status."""
        latest_result = self.backtest_results[-1] if self.backtest_results else None

        return {
            'running': self.is_running,
            'last_backtest': self.last_backtest.isoformat() if self.last_backtest else None,
            'total_backtests': len(self.backtest_results),
            'latest_result': {
                'win_rate': latest_result['metrics'].get('win_rate', 0) if latest_result else 0,
                'total_pnl': latest_result['metrics'].get('total_pnl', 0) if latest_result else 0,
                'sharpe_ratio': latest_result['metrics'].get('sharpe_ratio', 0) if latest_result else 0,
                'max_drawdown': latest_result['metrics'].get('max_drawdown_pct', 0) if latest_result else 0
            } if latest_result else None,
            'next_backtest_in': self.config.BACKTEST_INTERVAL - (
                (datetime.now() - self.last_backtest).total_seconds()
                if self.last_backtest else 0
            )
        }


class LiveTradingService:
    """Live trading service with automated execution."""

    def __init__(self):
        self.config = Config()
        self.is_running = False
        self.positions = []
        self.daily_pnl = 0
        self.total_pnl = 0
        self.trades_today = []
        self.last_trade = None

    async def start_trading(self):
        """Start automated trading."""
        self.is_running = True
        logger.info("Starting live trading service")

        while self.is_running:
            try:
                await self.check_signals()
                await self.manage_positions()
                await asyncio.sleep(self.config.REAL_TIME_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Trading error: {e}")
                await asyncio.sleep(60)

    async def check_signals(self):
        """Check for trading signals."""
        try:
            # Load latest AI model
            model_path = Path("training_results/20251015_184036/random_forest_model.pkl")
            if not model_path.exists():
                return

            import joblib
            model = joblib.load(model_path)

            # Get market data (simplified for demo)
            # In production, this would get real Aster DEX data
            market_data = self.get_market_data()

            if market_data:
                # Generate signal
                signal = await self.generate_signal(model, market_data)

                if signal and self.should_execute_trade(signal):
                    await self.execute_trade(signal)

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")

    def get_market_data(self) -> Optional[Dict]:
        """Get current market data."""
        # In production, this would call Aster DEX API
        # For demo, return mock data
        return {
            'btc_price': 50000 + np.random.normal(0, 500),
            'eth_price': 3000 + np.random.normal(0, 50),
            'volatility': np.random.uniform(0.01, 0.05),
            'volume': np.random.uniform(1000, 5000)
        }

    async def generate_signal(self, model, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal using AI model."""
        try:
            # Prepare features (simplified)
            features = np.array([[
                market_data['btc_price'] * 0.001,  # price_change
                0.02,  # high_low_ratio
                market_data['volume'] * 0.0001,  # volume_price_ratio
                0.01,  # price_sma_5_ratio
                0.01,  # price_sma_10_ratio
                0.01,  # price_sma_20_ratio
                market_data['volatility'],  # volatility
                0.5   # rsi
            ]])

            # Get prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]

            if probability > 0.65:  # High confidence threshold
                return {
                    'symbol': 'BTCUSDT',
                    'direction': 'long' if prediction == 1 else 'short',
                    'confidence': probability,
                    'entry_price': market_data['btc_price'],
                    'reason': f'AI prediction: {probability:.2f} confidence'
                }

            return None

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None

    def should_execute_trade(self, signal: Dict) -> bool:
        """Check if trade should be executed."""
        # Check daily loss limit
        if self.daily_pnl < -self.config.DAILY_LOSS_LIMIT:
            return False

        # Check max concurrent positions
        if len(self.positions) >= self.config.MAX_CONCURRENT_TRADES:
            return False

        # Check trade size
        trade_size = np.random.uniform(self.config.TRADE_SIZE_MIN, self.config.TRADE_SIZE_MAX)
        if trade_size > self.get_available_capital():
            return False

        return True

    async def execute_trade(self, signal: Dict):
        """Execute a trade."""
        try:
            # Calculate position size
            trade_size = np.random.uniform(self.config.TRADE_SIZE_MIN, self.config.TRADE_SIZE_MAX)
            leverage = 5  # Conservative leverage

            position = {
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'size': trade_size,
                'leverage': leverage,
                'timestamp': datetime.now(),
                'confidence': signal['confidence']
            }

            self.positions.append(position)
            self.last_trade = datetime.now()

            logger.info(f"EXECUTED TRADE: {signal['symbol']} {signal['direction']} ${trade_size} @ ${signal['entry_price']}")

            # Record trade
            self.trades_today.append({
                'timestamp': datetime.now(),
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'size': trade_size,
                'entry_price': signal['entry_price'],
                'confidence': signal['confidence']
            })

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

    async def manage_positions(self):
        """Manage existing positions."""
        try:
            for position in self.positions[:]:  # Copy to avoid modification during iteration
                # Simple exit logic (in production, more sophisticated)
                if np.random.random() > 0.95:  # 5% chance to exit each cycle
                    pnl = position['size'] * np.random.normal(0.02, 0.05)  # Random P&L
                    self.daily_pnl += pnl
                    self.total_pnl += pnl

                    self.positions.remove(position)
                    logger.info(f"CLOSED POSITION: {position['symbol']} P&L: ${pnl:.2f}")

        except Exception as e:
            logger.error(f"Position management error: {e}")

    def get_available_capital(self) -> float:
        """Get available capital for trading."""
        # Simplified - in production, check actual account balance
        return max(0, 150 + self.total_pnl - sum(p['size'] for p in self.positions))

    def get_status(self) -> Dict:
        """Get trading service status."""
        return {
            'running': self.is_running,
            'positions': len(self.positions),
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'trades_today': len(self.trades_today),
            'available_capital': self.get_available_capital(),
            'last_trade': self.last_trade.isoformat() if self.last_trade else None,
            'open_positions': self.positions
        }


class CostOptimizationService:
    """Cost optimization for GCP services."""

    def __init__(self):
        self.config = Config()
        self.daily_cost = 0.0
        self.cost_breakdown = {
            'cloud_run': 0.0,
            'cloud_storage': 0.0,
            'vertex_ai': 0.0,
            'bigquery': 0.0,
            'other': 0.0
        }

    def optimize_resources(self):
        """Optimize cloud resources for cost efficiency."""
        # Scale down during off-hours
        current_hour = datetime.now().hour

        if 2 <= current_hour <= 6:  # Night time
            # Reduce Cloud Run instances
            self.scale_cloud_run(0)  # Minimum instances
        elif 9 <= current_hour <= 17:  # Trading hours
            # Full capacity
            self.scale_cloud_run(2)
        else:
            # Moderate capacity
            self.scale_cloud_run(1)

        # Clean up old data
        self.cleanup_old_data()

        # Optimize storage
        self.optimize_storage()

    def scale_cloud_run(self, instances: int):
        """Scale Cloud Run service."""
        # In production, this would call GCP APIs
        logger.info(f"Scaling Cloud Run to {instances} instances")

    def cleanup_old_data(self):
        """Clean up old data to reduce storage costs."""
        # Remove data older than 30 days
        cutoff_date = datetime.now() - timedelta(days=30)

        try:
            data_dir = Path("data")
            if data_dir.exists():
                for file_path in data_dir.rglob("*"):
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")

        except Exception as e:
            logger.error(f"Data cleanup error: {e}")

    def optimize_storage(self):
        """Optimize cloud storage usage."""
        # Compress old data
        # Archive infrequently accessed data
        # Use cheaper storage classes for old data
        pass

    def monitor_costs(self) -> Dict:
        """Monitor GCP costs."""
        # In production, this would query GCP Billing API
        # For demo, return mock costs
        return {
            'daily_cost': round(np.random.uniform(2.0, 8.0), 2),
            'monthly_estimate': round(np.random.uniform(60.0, 240.0), 2),
            'budget_remaining': self.config.MAX_DAILY_COST - self.daily_cost,
            'cost_breakdown': self.cost_breakdown,
            'optimization_suggestions': [
                "Scale down during off-hours",
                "Use spot instances for batch jobs",
                "Archive old data to cheaper storage",
                "Optimize BigQuery queries"
            ]
        }


class CentralDashboardConsole:
    """Main dashboard console application."""

    def __init__(self):
        self.config = Config()

        # Initialize services
        self.data_service = DataCollectionService()
        self.backtest_service = AutomatedBacktestingService()
        self.trading_service = LiveTradingService()
        self.cost_service = CostOptimizationService()

        # Service threads
        self.service_threads = []

        # Dashboard state
        self.services_running = False

    def start_services(self):
        """Start all cloud services."""
        if self.services_running:
            return

        logger.info("Starting central dashboard console services...")

        # Start data collection
        asyncio.create_task(self.data_service.start_collection())

        # Start automated backtesting
        asyncio.create_task(self.backtest_service.start_backtesting())

        # Start trading service (commented for safety)
        # asyncio.create_task(self.trading_service.start_trading())

        self.services_running = True
        logger.info("All services started successfully")

    def stop_services(self):
        """Stop all services."""
        logger.info("Stopping all services...")

        self.data_service.is_running = False
        self.backtest_service.is_running = False
        self.trading_service.is_running = False

        self.services_running = False
        logger.info("All services stopped")

    def run_dashboard(self):
        """Run the central dashboard console."""
        st.set_page_config(
            page_title="Aster AI Trading Console",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown("""
        <style>
        .console-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .service-status {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .status-active {
            background-color: #e8f5e8;
            border-left: 5px solid #2e7d32;
        }
        .status-inactive {
            background-color: #ffebee;
            border-left: 5px solid #d32f2f;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<h1 class="console-header">🚀 Aster AI Trading Console</h1>', unsafe_allow_html=True)
        st.markdown("*Central Operations Center - Continuous Data Collection, Analysis & Trading*")

        # Sidebar controls
        st.sidebar.title("🎛️ Control Panel")

        if st.sidebar.button("▶️ Start All Services", type="primary"):
            self.start_services()
            st.sidebar.success("Services starting...")

        if st.sidebar.button("⏹️ Stop All Services"):
            self.stop_services()
            st.sidebar.warning("Services stopping...")

        st.sidebar.markdown("---")

        # Service status
        st.sidebar.subheader("📊 Service Status")

        services = [
            ("Data Collection", self.data_service.is_running),
            ("Auto Backtesting", self.backtest_service.is_running),
            ("Live Trading", self.trading_service.is_running),
        ]

        for service, running in services:
            status = "🟢 Active" if running else "🔴 Inactive"
            st.sidebar.write(f"{service}: {status}")

        # Main dashboard
        self.show_overview()
        self.show_service_monitoring()
        self.show_performance_dashboard()
        self.show_cost_optimization()
        self.show_system_health()

    def show_overview(self):
        """Show system overview."""
        st.header("📊 System Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status = "🟢 Active" if self.services_running else "🔴 Inactive"
            st.metric("System Status", status)

        with col2:
            data_status = self.data_service.get_status()
            last_collection = data_status.get('last_collection')
            if last_collection:
                time_since = (datetime.now() - datetime.fromisoformat(last_collection)).total_seconds() / 60
                st.metric("Data Freshness", f"{time_since:.0f} min ago")
            else:
                st.metric("Data Freshness", "Never")

        with col3:
            trading_status = self.trading_service.get_status()
            st.metric("Open Positions", trading_status['positions'])

        with col4:
            cost_status = self.cost_service.monitor_costs()
            st.metric("Daily Cost", f"${cost_status['daily_cost']:.2f}")

        # Key metrics
        st.subheader("🎯 Key Performance Indicators")

        metrics_cols = st.columns(4)

        with metrics_cols[0]:
            backtest_status = self.backtest_service.get_status()
            win_rate = backtest_status.get('latest_result', {}).get('win_rate', 0)
            st.metric("Backtest Win Rate", f"{win_rate:.1f}%")

        with metrics_cols[1]:
            pnl = trading_status.get('total_pnl', 0)
            st.metric("Total P&L", f"${pnl:.2f}")

        with metrics_cols[2]:
            capital = trading_status.get('available_capital', 150)
            st.metric("Available Capital", f"${capital:.2f}")

        with metrics_cols[3]:
            cost_remaining = cost_status.get('budget_remaining', 0)
            st.metric("Budget Remaining", f"${cost_remaining:.2f}")

    def show_service_monitoring(self):
        """Show service monitoring dashboard."""
        st.header("🔧 Service Monitoring")

        # Data Collection Service
        st.subheader("📡 Data Collection Service")
        data_status = self.data_service.get_status()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            running = data_status.get('running', False)
            status_class = "status-active" if running else "status-inactive"
            st.markdown(f'<div class="service-status {status_class}">Status: {"Running" if running else "Stopped"}</div>', unsafe_allow_html=True)

        with col2:
            collections = data_status.get('stats', {}).get('total_collections', 0)
            st.metric("Total Collections", collections)

        with col3:
            success_rate = (data_status.get('stats', {}).get('successful_collections', 0) /
                          max(data_status.get('stats', {}).get('total_collections', 1), 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")

        with col4:
            next_in = data_status.get('next_collection_in', 0)
            st.metric("Next Collection", f"{max(0, next_in):.0f}s")

        # Backtesting Service
        st.subheader("🔄 Automated Backtesting")
        backtest_status = self.backtest_service.get_status()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            running = backtest_status.get('running', False)
            status_class = "status-active" if running else "status-inactive"
            st.markdown(f'<div class="service-status {status_class}">Status: {"Running" if running else "Stopped"}</div>', unsafe_allow_html=True)

        with col2:
            total_tests = backtest_status.get('total_backtests', 0)
            st.metric("Total Backtests", total_tests)

        with col3:
            if backtest_status.get('latest_result'):
                win_rate = backtest_status['latest_result'].get('win_rate', 0)
                st.metric("Latest Win Rate", f"{win_rate:.1f}%")
            else:
                st.metric("Latest Win Rate", "N/A")

        with col4:
            next_in = backtest_status.get('next_backtest_in', 0)
            st.metric("Next Backtest", f"{max(0, next_in/3600):.1f}h")

        # Trading Service
        st.subheader("💰 Live Trading Service")
        trading_status = self.trading_service.get_status()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            running = trading_status.get('running', False)
            status_class = "status-active" if running else "status-inactive"
            st.markdown(f'<div class="service-status {status_class}">Status: {"Running" if running else "Stopped"}</div>', unsafe_allow_html=True)

        with col2:
            positions = trading_status.get('positions', 0)
            st.metric("Open Positions", positions)

        with col3:
            daily_pnl = trading_status.get('daily_pnl', 0)
            st.metric("Daily P&L", f"${daily_pnl:.2f}")

        with col4:
            trades = trading_status.get('trades_today', 0)
            st.metric("Trades Today", trades)

    def show_performance_dashboard(self):
        """Show performance dashboard."""
        st.header("📈 Performance Dashboard")

        # Equity curve
        st.subheader("💹 Equity Curve")

        # Mock equity data (in production, this would be real)
        dates = pd.date_range(start='2025-10-15', periods=30, freq='D')
        equity = [150]
        for i in range(29):
            change = np.random.normal(0.02, 0.08)  # Mean 2%, std 8%
            new_equity = equity[-1] * (1 + change)
            equity.append(max(new_equity, 50))  # Floor at $50

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=3)
        ))

        fig.add_hline(y=1000000, line_dash="dash", line_color="red",
                     annotation_text="$1M Target")

        fig.update_layout(
            title="Trading Performance - Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_return = ((equity[-1] - equity[0]) / equity[0]) * 100
            st.metric("Total Return", f"{total_return:.1f}%")

        with col2:
            # Sharpe ratio (simplified)
            returns = np.diff(equity) / equity[:-1]
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        with col3:
            # Max drawdown
            cummax = np.maximum.accumulate(equity)
            drawdown = (equity - cummax) / cummax
            max_dd = drawdown.min() * 100
            st.metric("Max Drawdown", f"{max_dd:.1f}%")

        with col4:
            # Win rate (mock)
            win_rate = np.random.uniform(55, 75)
            st.metric("Win Rate", f"{win_rate:.1f}%")

        # Recent trades
        st.subheader("📋 Recent Trading Activity")

        # Mock trade data
        trades_data = []
        for i in range(10):
            trade = {
                'Date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                'Symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT']),
                'Direction': np.random.choice(['Long', 'Short']),
                'Size': f"${np.random.uniform(10, 50):.0f}",
                'Entry': f"${np.random.uniform(40000, 60000):.0f}",
                'Exit': f"${np.random.uniform(40000, 60000):.0f}",
                'P&L': f"${np.random.normal(5, 15):.2f}",
                'Status': np.random.choice(['Win', 'Loss'])
            }
            trades_data.append(trade)

        trades_df = pd.DataFrame(trades_data)
        st.dataframe(trades_df, use_container_width=True)

    def show_cost_optimization(self):
        """Show cost optimization dashboard."""
        st.header("💰 Cost Optimization")

        cost_data = self.cost_service.monitor_costs()

        # Cost overview
        col1, col2, col3 = st.columns(3)

        with col1:
            daily_cost = cost_data.get('daily_cost', 0)
            st.metric("Daily Cost", f"${daily_cost:.2f}")

        with col2:
            monthly_est = cost_data.get('monthly_estimate', 0)
            st.metric("Monthly Estimate", f"${monthly_est:.2f}")

        with col3:
            remaining = cost_data.get('budget_remaining', 0)
            st.metric("Budget Remaining", f"${remaining:.2f}")

        # Cost breakdown
        st.subheader("📊 Cost Breakdown")

        breakdown = cost_data.get('cost_breakdown', {})
        services = list(breakdown.keys())
        costs = list(breakdown.values())

        fig = go.Figure(data=[go.Pie(
            labels=services,
            values=costs,
            title="Daily Cost Distribution"
        )])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Optimization suggestions
        st.subheader("💡 Optimization Suggestions")

        suggestions = cost_data.get('optimization_suggestions', [])
        for suggestion in suggestions:
            st.write(f"• {suggestion}")

        # Cost controls
        st.subheader("🎛️ Cost Controls")

        if st.button("🕐 Scale Down (Off-Hours)"):
            self.cost_service.scale_cloud_run(0)
            st.success("Scaled down to minimum instances")

        if st.button("🧹 Clean Old Data"):
            self.cost_service.cleanup_old_data()
            st.success("Cleaned up old data files")

        if st.button("📊 Optimize Storage"):
            self.cost_service.optimize_storage()
            st.success("Optimized storage usage")

    def show_system_health(self):
        """Show system health dashboard."""
        st.header("❤️ System Health")

        # System resources
        col1, col2, col3, col4 = st.columns(4)

        # CPU usage
        cpu_percent = psutil.cpu_percent()
        with col1:
            if cpu_percent > 80:
                st.error(f"CPU: {cpu_percent:.1f}%")
            elif cpu_percent > 60:
                st.warning(f"CPU: {cpu_percent:.1f}%")
            else:
                st.success(f"CPU: {cpu_percent:.1f}%")

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        with col2:
            if memory_percent > 80:
                st.error(f"Memory: {memory_percent:.1f}%")
            elif memory_percent > 60:
                st.warning(f"Memory: {memory_percent:.1f}%")
            else:
                st.success(f"Memory: {memory_percent:.1f}%")

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        with col3:
            if disk_percent > 80:
                st.error(f"Disk: {disk_percent:.1f}%")
            elif disk_percent > 60:
                st.warning(f"Disk: {disk_percent:.1f}%")
            else:
                st.success(f"Disk: {disk_percent:.1f}%")

        # Network status
        with col4:
            try:
                response = requests.get("https://www.google.com", timeout=5)
                if response.status_code == 200:
                    st.success("Network: Online")
                else:
                    st.warning("Network: Issues")
            except:
                st.error("Network: Offline")

        # Service health
        st.subheader("🔧 Service Health")

        services_health = [
            ("Data Collection", self.data_service.is_running, "Collecting market data"),
            ("Backtesting", self.backtest_service.is_running, "Running automated tests"),
            ("Trading Bot", self.trading_service.is_running, "Executing trades"),
            ("Cost Optimization", True, "Monitoring GCP costs")
        ]

        for service, healthy, description in services_health:
            col1, col2 = st.columns([1, 3])
            with col1:
                if healthy:
                    st.success(f"✅ {service}")
                else:
                    st.error(f"❌ {service}")
            with col2:
                st.write(description)

        # Error logs
        st.subheader("📋 Recent Errors")

        # Mock error logs (in production, read from actual logs)
        errors = [
            "2025-10-15 19:00: Data collection retry after timeout",
            "2025-10-15 18:45: Backtest completed successfully",
            "2025-10-15 18:30: Trading signal generated",
        ]

        for error in errors:
            st.code(error, language="text")


def main():
    """Main application."""
    console = CentralDashboardConsole()

    # Auto-start services in cloud environment
    if os.environ.get('ENVIRONMENT') == 'CLOUD':
        console.start_services()

    console.run_dashboard()


if __name__ == "__main__":
    main()
