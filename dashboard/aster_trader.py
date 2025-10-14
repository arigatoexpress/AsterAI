"""
Aster Autonomous Trader Dashboard
Simplified dashboard for monitoring and controlling the Aster DEX trading system.
"""

import sys
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import time
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_trader.config import get_settings
from mcp_trader.trading.autonomous_trader import AutonomousTrader, TradingMode
from mcp_trader.data.aster_feed import AsterDataFeed
from mcp_trader.data import get_crypto_prices


class AsterTraderDashboard:
    """Simplified dashboard for Aster autonomous trading."""

    def __init__(self):
        self.trader = None
        self.data_feed = AsterDataFeed()
        self.settings = get_settings()

        # Dashboard state
        self.is_running = False
        self.status_thread = None

    def run(self):
        """Run the dashboard."""
        st.set_page_config(
            page_title="🚀 Aster Autonomous Trader",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🚀 Aster Autonomous Trader")
        st.markdown("*AI-powered autonomous trading on Aster DEX*")

        # Sidebar controls
        self.render_sidebar()

        # Main dashboard
        self.render_main_dashboard()

    def render_sidebar(self):
        """Render sidebar controls."""
        with st.sidebar:
            st.header("🎛️ Control Panel")

            # Trading mode selection
            trading_mode = st.selectbox(
                "Trading Mode",
                ["grid", "volatility", "hybrid"],
                index=0,
                help="Select autonomous trading strategy"
            )

            # Risk settings
            st.subheader("🛡️ Risk Settings")
            max_positions = st.slider("Max Concurrent Positions", 1, 10, 3)
            max_portfolio_risk = st.slider("Max Portfolio Risk %", 1, 20, 10)

            # Control buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("▶️ START", type="primary", use_container_width=True):
                    self.start_trading(trading_mode, max_positions, max_portfolio_risk)

            with col2:
                if st.button("⏹️ STOP", type="secondary", use_container_width=True):
                    self.stop_trading()

            # Emergency controls
            st.subheader("🚨 Emergency")
            if st.button("🚨 EMERGENCY STOP", type="secondary", use_container_width=True):
                self.emergency_stop()

            # System status
            st.subheader("📊 System Status")
            status_placeholder = st.empty()

            # Update status periodically
            if self.is_running:
                self.update_status_display(status_placeholder)

    def render_main_dashboard(self):
        """Render main dashboard content."""
        # Portfolio overview
        self.render_portfolio_overview()

        # Active positions
        self.render_active_positions()

        # Market overview
        self.render_market_overview()

        # Trading performance
        self.render_performance_metrics()

        # Recent trades
        self.render_recent_trades()

    def render_portfolio_overview(self):
        """Render portfolio overview section."""
        st.header("💰 Portfolio Overview")

        if not self.trader:
            st.info("🤖 Trading system not initialized. Start trading to see portfolio data.")
            return

        portfolio_status = self.trader.get_portfolio_status()

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Balance",
                f"${portfolio_status['total_balance']:.2f}",
                delta=f"{portfolio_status.get('daily_change', 0):+.2f}%"
            )

        with col2:
            st.metric(
                "Unrealized P&L",
                f"${portfolio_status['unrealized_pnl']:.2f}",
                delta=f"{portfolio_status.get('pnl_change', 0):+.2f}%"
            )

        with col3:
            st.metric(
                "Active Positions",
                portfolio_status['active_positions']
            )

        with col4:
            st.metric(
                "Active Grids",
                portfolio_status['active_grids']
            )

        # Portfolio allocation chart
        if portfolio_status['active_positions'] > 0:
            st.subheader("Portfolio Allocation")

            # Mock data for now - would come from actual positions
            allocation_data = {
                'BTCUSDT': 40,
                'ETHUSDT': 30,
                'SOLUSDT': 20,
                'Others': 10
            }

            fig = go.Figure(data=[go.Pie(
                labels=list(allocation_data.keys()),
                values=list(allocation_data.values()),
                hole=.3
            )])

            fig.update_layout(
                title="Asset Allocation",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff"
            )

            st.plotly_chart(fig, use_container_width=True)

    def render_active_positions(self):
        """Render active positions table."""
        st.header("📈 Active Positions")

        if not self.trader:
            st.info("No active positions.")
            return

        # Mock active positions data
        positions_data = [
            {
                'Symbol': 'BTCUSDT',
                'Side': 'LONG',
                'Quantity': 0.05,
                'Entry Price': 45000,
                'Current Price': 45200,
                'P&L': 50.00,
                'P&L %': 2.22
            },
            {
                'Symbol': 'ETHUSDT',
                'Side': 'SHORT',
                'Quantity': 1.2,
                'Entry Price': 2800,
                'Current Price': 2750,
                'P&L': 60.00,
                'P&L %': 1.79
            }
        ]

        if positions_data:
            df = pd.DataFrame(positions_data)
            st.dataframe(df, use_container_width=True)

            # Position performance chart
            pnl_data = [pos['P&L'] for pos in positions_data]
            symbols = [pos['Symbol'] for pos in positions_data]

            fig = go.Figure(data=[go.Bar(
                x=symbols,
                y=pnl_data,
                marker_color=['green' if x > 0 else 'red' for x in pnl_data]
            )])

            fig.update_layout(
                title="Position P&L",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active positions.")

    def render_market_overview(self):
        """Render market overview section."""
        st.header("🌍 Market Overview")

        # Get live prices
        prices = get_crypto_prices(['BTC', 'ETH', 'SOL', 'SUI', 'PENGU', 'ASTER'])

        if prices:
            # Create price grid
            cols = st.columns(3)
            price_items = []

            for symbol, data in prices.items():
                if isinstance(data, dict) and 'price' in data:
                    price = data['price']
                    change = data.get('change_24h', 0)
                    price_items.append((symbol, price, change))

            for i, (symbol, price, change) in enumerate(price_items):
                with cols[i % 3]:
                    delta_color = "normal" if change >= 0 else "inverse"
                    st.metric(
                        symbol,
                        f"${price:.2f}" if price >= 1 else f"${price:.6f}",
                        f"{change:+.2f}%" if change is not None else "N/A",
                        delta_color=delta_color
                    )

            # Market regime indicator
            st.subheader("📊 Market Regime")
            regime_col1, regime_col2, regime_col3 = st.columns(3)

            with regime_col1:
                st.metric("BTC Volatility", "2.4%", "+0.3%")

            with regime_col2:
                st.metric("ETH Momentum", "Bullish", "↗️")

            with regime_col3:
                st.metric("Overall Regime", "High Volatility", "⚠️")

    def render_performance_metrics(self):
        """Render trading performance metrics."""
        st.header("📊 Performance Metrics")

        # Performance metrics grid
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Return", "+5.2%", "+0.8% today")

        with col2:
            st.metric("Win Rate", "68%", "+5% this week")

        with col3:
            st.metric("Sharpe Ratio", "2.1", "+0.2")

        with col4:
            st.metric("Max Drawdown", "-3.2%", "improving")

        # Performance chart
        st.subheader("Portfolio Performance")

        # Mock performance data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        portfolio_values = [10000 * (1 + 0.001 * i + 0.0001 * np.random.randn()) for i in range(len(dates))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff88', width=2)
        ))

        fig.update_layout(
            title="Portfolio Value Over Time",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
            xaxis=dict(gridcolor="#374151"),
            yaxis=dict(gridcolor="#374151")
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_recent_trades(self):
        """Render recent trades table."""
        st.header("💹 Recent Trades")

        # Mock recent trades data
        trades_data = [
            {
                'Time': datetime.now() - timedelta(minutes=5),
                'Symbol': 'BTCUSDT',
                'Side': 'BUY',
                'Quantity': 0.02,
                'Price': 45100,
                'Reason': 'Grid level buy'
            },
            {
                'Time': datetime.now() - timedelta(minutes=15),
                'Symbol': 'ETHUSDT',
                'Side': 'SELL',
                'Quantity': 0.8,
                'Price': 2820,
                'Reason': 'Profit taking'
            },
            {
                'Time': datetime.now() - timedelta(hours=1),
                'Symbol': 'SOLUSDT',
                'Side': 'BUY',
                'Quantity': 50,
                'Price': 98.5,
                'Reason': 'Volatility signal'
            }
        ]

        df = pd.DataFrame(trades_data)
        df['Time'] = df['Time'].dt.strftime('%H:%M:%S')
        st.dataframe(df, use_container_width=True)

    def start_trading(self, mode: str, max_positions: int, max_risk: float):
        """Start the autonomous trading system."""
        try:
            config = {
                'trading_mode': mode,
                'max_concurrent_positions': max_positions,
                'risk_config': {
                    'limits': {
                        'max_portfolio_risk': max_risk / 100.0
                    }
                }
            }

            self.trader = AutonomousTrader(config)
            asyncio.run(self.trader.start())
            self.is_running = True

            st.success(f"🚀 Trading started in {mode} mode!")

        except Exception as e:
            st.error(f"Failed to start trading: {e}")

    def stop_trading(self):
        """Stop the autonomous trading system."""
        if self.trader:
            asyncio.run(self.trader.stop())
            self.is_running = False
            st.success("⏹️ Trading stopped successfully!")

    def emergency_stop(self):
        """Emergency stop all trading."""
        if self.trader:
            self.trader.set_emergency_stop(True)
            st.error("🚨 Emergency stop activated!")

    def update_status_display(self, placeholder):
        """Update status display in sidebar."""
        if self.trader and self.is_running:
            status = self.trader.get_portfolio_status()
            placeholder.markdown(f"""
            **Status:** 🟢 Running
            **Positions:** {status['active_positions']}
            **Balance:** ${status['total_balance']:.2f}
            **P&L:** ${status['unrealized_pnl']:+.2f}
            """)
        else:
            placeholder.markdown("**Status:** 🔴 Stopped")


def main():
    """Main dashboard entry point."""
    dashboard = AsterTraderDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
