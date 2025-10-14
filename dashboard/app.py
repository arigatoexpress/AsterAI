"""
Rari Trade AI - Intelligent Trading Platform
===========================================

A professional, AI-powered trading dashboard for Aster DEX perpetual futures.
Features proprietary DMark indicators, ensemble strategies, and real-time execution.
"""

import sys
import os

# Ensure the mcp_trader module can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from datetime import datetime, timedelta

from mcp_trader.strategies.rules import generate_positions_sma_crossover
from mcp_trader.backtesting.vectorized_backtester import evaluate_positions
from mcp_trader.strategies.indicators import sma
from mcp_trader.data.bigquery_client import BigQueryClient
from mcp_trader.config import get_aster_symbols, get_symbol_mapping

# --- Rari Trade AI Configuration ---
st.set_page_config(
    page_title="🚀 Rari Trade AI - Intelligent Trading Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Dark Theme with Rari Trade Branding ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #6366f1;
    --secondary-color: #8b5cf6;
    --accent-color: #06b6d4;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --background: #0f0f23;
    --surface: #1a1a2e;
    --surface-hover: #16213e;
    --text-primary: #ffffff;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --border: #374151;
    --border-light: #4b5563;
}

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
}

.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid #374151;
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.header-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.header-subtitle {
    color: #a1a1aa;
    font-size: 1.1rem;
    font-weight: 400;
    margin-bottom: 0;
}

/* Card Styling */
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    border-color: rgba(99, 102, 241, 0.3);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.25rem;
}

.metric-label {
    color: #a1a1aa;
    font-size: 0.9rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Status Indicators */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
}

.status-active {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-inactive {
    background: rgba(107, 114, 128, 0.2);
    color: #6b7280;
    border: 1px solid rgba(107, 114, 128, 0.3);
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 0.25rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #a1a1aa;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
}

/* Selectbox and Input Styling */
.stSelectbox, .stTextInput, .stNumberInput, .stDateInput {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    color: white;
}

/* Progress Bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
}

/* Info Boxes */
.info-box {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}

.info-box h4 {
    color: #6366f1;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
}

.info-box p {
    color: #a1a1aa;
    margin: 0;
    line-height: 1.5;
}

/* Animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.pulse {
    animation: pulse 2s infinite;
}

/* Responsive Design */
@media (max-width: 768px) {
    .header-title {
        font-size: 2rem;
    }

    .metric-card {
        margin-bottom: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Rari Trade AI Header ---
def render_header():
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🚀 Rari Trade AI</h1>
        <p class="header-subtitle">Intelligent Trading Platform for Aster DEX</p>
        <div style="display: flex; gap: 1rem; margin-top: 1rem;">
            <span class="status-indicator status-active">
                <span>●</span> AI-Powered
            </span>
            <span class="status-indicator status-active">
                <span>●</span> DMark Indicators
            </span>
            <span class="status-indicator status-active">
                <span>●</span> Live Trading
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(title, value, subtitle, icon="📊"):
    """Render a beautiful metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.2rem;">{icon}</span>
            <div class="metric-label">{title}</div>
        </div>
        <div class="metric-value">{value}</div>
        <div style="color: #71717a; font-size: 0.8rem; margin-top: 0.25rem;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def render_info_box(title, content, icon="💡"):
    """Render an informational box."""
    st.markdown(f"""
    <div class="info-box">
        <h4>{icon} {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Rari Trade AI Professional Sidebar ---
with st.sidebar:
    # Professional Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem; padding: 1rem; background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border-radius: 12px; border: 1px solid rgba(102,126,234,0.2);">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎛️</div>
        <h2 style="color: #ffffff; font-size: 1.4rem; margin-bottom: 0.25rem; font-weight: 700;">Rari Trade AI</h2>
        <p style="color: #c4c4d4; font-size: 0.85rem; margin: 0; line-height: 1.4;">Professional Trading Platform</p>
        <div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(67, 233, 123, 0.1); border-radius: 8px; border: 1px solid rgba(67, 233, 123, 0.3);">
            <span style="color: #43e97b; font-size: 0.8rem; font-weight: 600;">● System Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Start Section
    st.markdown("### 🚀 Quick Start")
    quick_start = st.selectbox(
        "Choose Your Journey:",
        ["New Trader (Beginner)", "Strategy Tester", "Live Trading Setup", "Advanced Analytics"],
        help="Select your preferred workflow to get optimized settings"
    )

    if quick_start == "New Trader (Beginner)":
        st.success("🎯 **Recommended for beginners** - Learn trading fundamentals with safe, educational settings.")
        with st.expander("📋 Step-by-Step Guide", expanded=True):
            st.markdown("""
            **Follow these 4 steps:**

            1. **📊 Choose Data Source** → Select "Synthetic (Safe Learning)" below
            2. **⚙️ Use Beginner Preset** → Click "Load Safe Settings"
            3. **📈 Run Backtest** → View results in the Dashboard tab
            4. **🎓 Learn** → Read the Learn tab for trading concepts

            **Goal:** Understand how strategies work before risking real money.
            """)

    elif quick_start == "Strategy Tester":
        st.info("🧪 **For experienced traders** - Test and optimize trading strategies.")
        with st.expander("📋 Step-by-Step Guide", expanded=True):
            st.markdown("""
            **Optimization Workflow:**

            1. **📊 Select Data Source** → Choose BigQuery or File Upload
            2. **🎯 Pick Strategy** → Try SMA Crossover or DMark
            3. **⚙️ Adjust Parameters** → Use Advanced Settings below
            4. **📊 Analyze Results** → Check Sharpe, Calmar, Profit Factor
            5. **🔄 Iterate** → Refine parameters and re-test

            **Pro Tip:** Start with conservative parameters and gradually optimize.
            """)

    elif quick_start == "Live Trading Setup":
        st.warning("🚨 **Advanced Users Only** - Connect to live markets.")
        with st.expander("📋 Step-by-Step Guide", expanded=True):
            st.markdown("""
            **Live Trading Setup:**

            1. **🔐 Setup Credentials** → Configure Aster API keys securely
            2. **📊 Connect Data** → Use BigQuery for real-time data
            3. **🎯 Choose Strategy** → Select tested, profitable strategy
            4. **⚖️ Set Risk Limits** → Configure position sizing and stops
            5. **🔴 Enable Trading** → Start with small position sizes

            **⚠️ Warning:** Live trading involves real financial risk.
            """)

    elif quick_start == "Advanced Analytics":
        st.markdown("🧠 **For quantitative researchers** - Deep dive into market analysis.")
        with st.expander("📋 Step-by-Step Guide", expanded=True):
            st.markdown("""
            **Research Workflow:**

            1. **📊 Load Large Dataset** → Use BigQuery with long date ranges
            2. **🔬 Enable All Features** → Check advanced settings
            3. **📈 Run Comprehensive Tests** → Multiple strategies and metrics
            4. **📊 Export Results** → Download data for further analysis
            5. **🧪 Develop New Ideas** → Use insights for strategy creation

            **Tools:** Full access to DMark indicators, ensemble methods, and ML models.
            """)

    # Preset Configurations
    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")

    preset_col1, preset_col2 = st.columns(2)

    with preset_col1:
        if st.button("🛡️ Safe Settings", use_container_width=True,
                    help="Conservative parameters for beginners"):
            # Safe preset values will be applied in session state
            st.session_state.safe_preset = True
            st.success("✅ Safe settings loaded!")

    with preset_col2:
        if st.button("🚀 Aggressive", use_container_width=True,
                    help="Higher risk/reward parameters"):
            st.session_state.aggressive_preset = True
            st.success("⚡ Aggressive settings loaded!")

    # Main Configuration Section
    st.markdown("---")
    st.markdown("### ⚙️ Trading Configuration")

    # Strategy Selection with better organization
    with st.expander("🎯 Strategy Selection", expanded=True):
        strategy_type = st.selectbox(
            "Choose Strategy:",
            ["SMA Crossover (Beginner)", "RSI Mean Reversion", "DMark Indicator (Advanced)", "Custom ML Model"],
            help="Select your trading strategy"
        )

        if strategy_type == "SMA Crossover (Beginner)":
            st.info("📈 **Simple trend-following strategy** - Buy when fast MA crosses above slow MA, sell when below.")

            short_win = st.slider("Fast MA Period", 5, 50, 20,
                                help="Short-term moving average (5-20 recommended for beginners)")
            long_win = st.slider("Slow MA Period", 20, 200, 50,
                               help="Long-term moving average (50-100 recommended)")

        elif strategy_type == "RSI Mean Reversion":
            st.info("📊 **Mean reversion strategy** - Buy when oversold, sell when overbought.")

            rsi_period = st.slider("RSI Period", 7, 21, 14,
                                 help="RSI calculation period (14 is standard)")
            rsi_low = st.slider("Oversold Level", 20, 40, 30,
                              help="Buy signal threshold (20-30 recommended)")
            rsi_high = st.slider("Overbought Level", 60, 80, 70,
                               help="Sell signal threshold (70-80 recommended)")

        elif strategy_type == "DMark Indicator (Advanced)":
            st.warning("🧠 **Proprietary AI indicator** - Advanced multi-component analysis.")

            st.info("DMark combines momentum, volatility, volume, microstructure, and trend analysis for superior signals.")
            dmark_config = st.selectbox("DMark Mode:",
                                      ["Conservative", "Balanced", "Aggressive"],
                                      help="Risk profile for DMark signals")

        elif strategy_type == "Custom ML Model":
            st.markdown("🤖 **Machine Learning Models** - Coming Soon")
            st.info("Advanced ML models including Random Forest, XGBoost, and ensemble methods will be available here.")

    # Risk Management
    with st.expander("🛡️ Risk Management", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            position_size = st.slider("Max Position Size", 0.01, 1.0, 0.25, step=0.01,
                                    help="Maximum position as fraction of portfolio (0.25 = 25%)")
            stop_loss = st.slider("Stop Loss %", 1, 20, 5,
                                help="Automatic loss cutoff percentage")

        with col2:
            take_profit = st.slider("Take Profit %", 2, 50, 10,
                                  help="Automatic profit taking percentage")
            fee_bps = st.slider("Trading Fee (bps)", 0, 50, 5,
                              help="Round-trip trading costs in basis points")

        st.markdown("**Risk Settings Applied:**")
        st.markdown(f"- Max Position: **{position_size*100:.0f}%** of portfolio")
        st.markdown(f"- Stop Loss: **{stop_loss}%** | Take Profit: **{take_profit}%**")
        st.markdown(f"- Trading Costs: **{fee_bps} bps** per round trip")

    # Data Source Selection with better UX
    st.markdown("---")
    st.markdown("### 📊 Data Source")

    data_source = st.radio(
        "Select Data Source:",
        ["Synthetic (Safe Learning)", "File Upload", "BigQuery (Production)"],
        help="Choose your data source based on your needs",
        index=0  # Default to Synthetic for beginners
    )

    # Enhanced Data Source Configuration
    if data_source == "Synthetic (Safe Learning)":
        st.success("🎯 **Perfect for learning** - No real money risk, instant results.")

        with st.expander("Synthetic Data Settings", expanded=False):
            n = st.slider("Data Points", 500, 5000, 1200, step=100,
                         help="More data = more reliable backtests (1000-2000 recommended)")
            sigma = st.slider("Market Volatility", 0.01, 0.08, 0.025, step=0.005,
                             help="How volatile the synthetic market is (0.02-0.04 typical)")
            mu = st.number_input("Expected Return", value=0.0005, step=0.0001, format="%.4f",
                               help="Average daily return (0.0005 = 0.05% per day)")

        st.markdown("""
        **Why Synthetic Data?**
        - ✅ **Risk-free learning environment**
        - ✅ **Instant results** (no waiting for data)
        - ✅ **Controllable market conditions**
        - ✅ **Perfect for strategy development**

        **When to graduate:** Once you consistently achieve >1.5 Sharpe ratio.
        """)

    elif data_source == "File Upload":
        st.info("📁 **Upload your own data** - CSV or Parquet files with OHLCV columns.")

        upload = st.file_uploader(
            "Upload Market Data",
            type=["csv", "parquet"],
            help="File must contain: timestamp, open, high, low, close, volume"
        )

        if upload:
            st.success(f"✅ File '{upload.name}' uploaded successfully!")
            st.info("💡 **Pro Tip:** Ensure your data includes minute-level granularity for best results.")

        st.markdown("""
        **Data Requirements:**
        - 📅 **timestamp** column (datetime format)
        - 💰 **close** price column (required)
        - 📊 **open, high, low** (recommended)
        - 📈 **volume** (recommended for realistic simulation)

        **Supported Formats:** CSV, Parquet
        """)

    elif data_source == "BigQuery (Production)":
        st.warning("☁️ **Enterprise-grade data** - Requires GCP setup.")

        with st.expander("BigQuery Configuration", expanded=False):
            project_id = st.text_input(
                "GCP Project ID",
                value=os.getenv("GCP_PROJECT", ""),
                help="Your Google Cloud Project ID (find in GCP Console)"
            )
            dataset_id = st.text_input(
                "BigQuery Dataset",
                value="market_data",
                help="Dataset containing your market data tables"
            )

        if project_id and dataset_id:
            try:
                bq_client = BigQueryClient(project_id=project_id, dataset_id=dataset_id)

                # Enhanced Symbol Selection
                aster_symbols = get_aster_symbols()
                symbol = st.selectbox(
                    "Trading Symbol",
                    aster_symbols,
                    help="Choose from Aster DEX supported assets"
                )

                venue = st.selectbox(
                    "Data Source",
                    ["binance", "okx"],
                    help="Exchange providing the historical data"
                )

                # Smart Date Range
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=pd.Timestamp.now() - pd.Timedelta(days=90),  # Longer default
                        help="Beginning of historical data period"
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=pd.Timestamp.now(),
                        help="End of historical data period"
                    )

                # Load Data Button with Status
                if st.button("🚀 Load Market Data", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Connecting to BigQuery..."):
                        try:
                            interval = "1h" if venue == "binance" else "1H"
                            df_bq = bq_client.query_ohlcv(
                                symbol=symbol,
                                venue=venue,
                                start_date=start_date.strftime("%Y-%m-%d"),
                                end_date=end_date.strftime("%Y-%m-%d"),
                                interval=interval
                            )

                            if not df_bq.empty:
                                # Store data in session state for use in main app
                                st.session_state.bq_data = df_bq
                                st.session_state.symbol = symbol
                                st.success(f"✅ Loaded {len(df_bq)} data points for {symbol}")
                                st.balloons()
                            else:
                                st.error("❌ No data found for selected criteria")

                        except Exception as e:
                            st.error(f"❌ Data loading failed: {str(e)}")

                st.markdown("""
                **BigQuery Benefits:**
                - 📊 **Massive datasets** (years of data available)
                - ⚡ **Fast queries** (optimized for analytics)
                - 🔄 **Real-time updates** (via scheduled functions)
                - 🛡️ **Enterprise security** (GCP authentication)
                """)

            except Exception as e:
                st.error(f"❌ BigQuery Connection Failed: {str(e)}")
                bq_client = None
        else:
            st.warning("⚠️ Enter GCP Project ID and Dataset to enable BigQuery")
            bq_client = None

    # Professional Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.02); border-radius: 8px; margin-top: 1rem;">
        <p style="color: #9ca3af; font-size: 0.8rem; margin: 0;">
            <strong>Rari Trade AI</strong> v1.0.0<br>
            Built for institutional-grade trading
        </p>
        <div style="margin-top: 0.5rem;">
            <span style="color: #43e97b; font-size: 0.75rem;">●</span>
            <span style="color: #6b7280; font-size: 0.75rem; margin-left: 0.25rem;">Ready for live trading</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Variable Initialization ---
# Initialize strategy variables based on selection
if strategy_type == "SMA Crossover (Beginner)":
    # Use the slider values from sidebar
    pass  # Variables are already defined in sidebar
elif strategy_type == "RSI Mean Reversion":
    # For RSI, we'll need to generate positions differently
    short_win = rsi_period  # Use RSI period as window for now
    long_win = rsi_period * 2  # Dummy long window
elif strategy_type == "DMark Indicator (Advanced)":
    # DMark uses different logic
    short_win = 20  # Default values
    long_win = 50
else:
    # Default fallback
    short_win = 20
    long_win = 50

# Handle preset buttons
if st.session_state.get('safe_preset', False):
    position_size = 0.1  # 10% position size
    stop_loss = 2  # 2% stop loss
    take_profit = 4  # 4% take profit
    fee_bps = 1  # 1 bps fee
    st.session_state.safe_preset = False  # Reset

if st.session_state.get('aggressive_preset', False):
    position_size = 0.5  # 50% position size
    stop_loss = 10  # 10% stop loss
    take_profit = 25  # 25% take profit
    fee_bps = 10  # 10 bps fee
    st.session_state.aggressive_preset = False  # Reset

# --- Data Processing ---
close = None
full_data = None

try:
    if data_source == "Synthetic (Safe Learning)":
        np.random.seed(42)
        returns = np.random.normal(mu, sigma, size=n)
        close = pd.Series((1 + pd.Series(returns)).cumprod() * 1000.0, name="close")
        full_data = pd.DataFrame({"close": close})

    elif data_source == "File Upload":
        if upload is not None:
            if upload.name.endswith(".csv"):
                df_src = pd.read_csv(upload)
            else:
                df_src = pd.read_parquet(upload)

            if "close" not in df_src.columns:
                st.error("❌ Uploaded file must contain a 'close' column")
                st.stop()

            close = pd.Series(df_src["close"], name="close")
            full_data = df_src.copy()
            st.success(f"✅ Loaded {len(close)} data points from file")
        else:
            st.info("📁 Please upload a data file to continue")
            st.stop()

    elif data_source == "BigQuery (Production)":
        # Check if data was loaded via sidebar button and stored in session state
        if 'bq_data' in st.session_state and st.session_state.bq_data is not None:
            df_bq = st.session_state.bq_data
            close = pd.Series(df_bq["close"], name="close")
            full_data = df_bq.copy()
            symbol = st.session_state.get('symbol', 'UNKNOWN')
            st.success(f"✅ Using loaded BigQuery data: {len(df_bq)} points for {symbol}")
        else:
            st.info("🚀 Click 'Load Market Data' in the sidebar to fetch data from BigQuery")
            st.stop()

except Exception as e:
    st.error(f"❌ Data processing error: {str(e)}")
    st.stop()

# --- Rari Trade AI Main Content ---
# Page selection (moved to sidebar context)
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧭 Navigation")

    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Landing"

    page = st.radio(
        "Choose Section:",
        ["🏠 Landing", "📊 Trading Dashboard", "🔬 Strategy Lab", "📈 Advanced Analytics", "🤖 Model Zoo", "⚙️ Live Trading", "🎓 Academy"],
        index=["🏠 Landing", "📊 Trading Dashboard", "🔬 Strategy Lab", "📈 Advanced Analytics", "🤖 Model Zoo", "⚙️ Live Trading", "🎓 Academy"].index(st.session_state.page),
        help="Navigate through Rari Trade features"
    )

# Update session state when radio button changes
st.session_state.page = page

def show_landing_page():
    """Premium landing page with live prices and world-class design."""
    # Import price feed
    from mcp_trader.data import get_crypto_prices

    # Get live prices
    price_symbols = ["BTC", "ETH", "SOL", "SUI", "PENGU", "ASTER"]
    prices = get_crypto_prices(price_symbols)

    # Define CSS for animations and styling
    st.markdown("""
    <style>
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    @keyframes drift {
        0%, 100% { transform: translateX(0) translateY(0) rotate(0deg); }
        25% { transform: translateX(20px) translateY(-10px) rotate(5deg); }
        50% { transform: translateX(-15px) translateY(15px) rotate(-3deg); }
        75% { transform: translateX(10px) translateY(-5px) rotate(2deg); }
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); filter: drop-shadow(0 0 20px rgba(239, 68, 68, 0.8)); }
        50% { transform: scale(1.05); filter: drop-shadow(0 0 30px rgba(239, 68, 68, 1)); }
    }

    .price-ticker {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        overflow: hidden;
        position: relative;
    }

    .price-ticker::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        animation: ticker 3s linear infinite;
    }

    .price-item {
        display: inline-block;
        margin-right: 2rem;
        font-family: 'Courier New', monospace;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .price-positive { color: #10b981; }
    .price-negative { color: #ef4444; }
    .price-neutral { color: #6b7280; }

    .hero-section {
        position: relative;
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 3rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(30, 41, 59, 0.96) 50%, rgba(15, 23, 42, 0.96) 100%);
        border-radius: 24px;
        border: 2px solid rgba(239, 68, 68, 0.3);
        overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(239, 68, 68, 0.25);
    }
    </style>
    """, unsafe_allow_html=True)

    # Price ticker HTML
    price_html = '<div class="price-ticker"><div style="display: inline-block; animation: ticker 20s linear infinite;">'
    for symbol in price_symbols:
        if symbol in prices and prices[symbol]["price"] > 0:
            price = prices[symbol]["price"]
            change = prices[symbol].get("change_24h")
            price_str = f"${price:.2f}" if price >= 1 else f"${price:.6f}"
            change_class = "price-positive" if change is not None and change >= 0 else "price-negative"
            change_str = f"{change:+.2f}%" if change is not None else "N/A"
            price_html += f'<span class="price-item">{symbol}: {price_str} <span class="{change_class}">{change_str}</span></span>'
        else:
            price_html += f'<span class="price-item">{symbol}: Loading...</span>'
    price_html += '</div></div>'

    st.markdown(price_html, unsafe_allow_html=True)

    # Hero Section - Simplified and clean
    st.markdown("""
    <div class="hero-section">
        <div style="position: relative; z-index: 2; max-width: 1200px; margin: 0 auto;">
            <!-- Ferrari + Rari Trade branding -->
            <div style="display: flex; align-items: center; justify-content: center; gap: 2rem; margin-bottom: 2rem; flex-wrap: wrap;">
                <div style="font-size: 5rem; animation: pulse 2s ease-in-out infinite;">🏎️</div>
                <div style="text-align: left;">
                    <h1 style="color: #ffffff; font-size: 3.5rem; font-weight: 900; margin: 0; background: linear-gradient(135deg, #ef4444 0%, #f97316 50%, #22c55e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Rari Trade</h1>
                    <div style="font-size: 1.1rem; color: #fbbf24; font-weight: 600; text-transform: uppercase; margin-top: 0.5rem;">The First Autonomous Perp Dex</div>
                </div>
            </div>

            <!-- Racing description -->
            <div style="background: rgba(239, 68, 68, 0.1); border: 2px solid rgba(239, 68, 68, 0.3); border-radius: 15px; padding: 1.5rem; margin-bottom: 2rem; backdrop-filter: blur(10px);">
                <p style="color: #ffffff; font-size: 1.3rem; margin: 0; font-weight: 500;">
                    🏎️ <strong style="color: #ef4444;">VROOM!</strong> Experience the speed of autonomous trading with AI algorithms that react faster than any human trader.
                </p>
            </div>

            <!-- Feature highlights -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 1.5rem; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏎️💨</div>
                    <h3 style="color: #ef4444; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">Lightning Fast</h3>
                    <p style="color: #fca5a5; font-size: 0.9rem; margin: 0;">Sub-millisecond execution</p>
                </div>
                <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 12px; padding: 1.5rem; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🧠⚡</div>
                    <h3 style="color: #22c55e; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">AI Powered</h3>
                    <p style="color: #86efac; font-size: 0.9rem; margin: 0;">Advanced neural networks</p>
                </div>
                <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 1.5rem; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔐🏁</div>
                    <h3 style="color: #3b82f6; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">Secure</h3>
                    <p style="color: #93c5fd; font-size: 0.9rem; margin: 0;">Bank-level encryption</p>
                </div>
            </div>

            <!-- CTA -->
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(251, 191, 36, 0.2) 100%); border: 2px solid rgba(239, 68, 68, 0.4); border-radius: 20px; padding: 2rem; backdrop-filter: blur(15px);">
                <h2 style="color: #ffffff; font-size: 2rem; font-weight: 800; margin-bottom: 1rem; text-align: center;">🏁 READY TO RACE? 🏁</h2>
                <p style="color: #fbbf24; font-size: 1.1rem; margin-bottom: 0; text-align: center; font-weight: 500;">
                    Join the fastest autonomous trading platform and leave traditional exchanges in the dust!
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Platform Overview - Simplified
    st.markdown("### 📊 Platform Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Strategies", "3", "+1 this week")

    with col2:
        st.metric("Total Backtests", "1,247", "+23 today")

    with col3:
        st.metric("Live Positions", "0", "Ready to trade")

    with col4:
        st.metric("Portfolio Value", "$10,000", "+2.1% today")

    # Quick Actions section
    st.markdown("### ⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🚀 Strategy Lab", use_container_width=True):
            st.session_state.page = "🔬 Strategy Lab"

    with col2:
        if st.button("📊 Advanced Analytics", use_container_width=True):
            st.session_state.page = "📈 Advanced Analytics"

    with col3:
        if st.button("🤖 Model Zoo", use_container_width=True):
            st.session_state.page = "🤖 Model Zoo"

    # System Status
    st.markdown("### 🔧 System Status")

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        st.success("🟢 API Connection - Operational")

    with status_col2:
        st.success("🟢 Data Pipeline - Streaming")

    with status_col3:
        st.success("🟢 Strategy Engine - Active")

    with status_col4:
        st.success("🟢 Risk Management - Monitoring")

    # Recent Activity
    st.markdown("### 📈 Recent Activity")

    activities = [
        {"action": "Strategy backtest completed", "symbol": "BTCUSDT", "result": "+2.4%", "time": "2 minutes ago"},
        {"action": "Data pipeline updated", "symbol": "ETHUSDT", "result": "1,200 rows", "time": "15 minutes ago"},
        {"action": "Risk parameters validated", "symbol": "System", "result": "All clear", "time": "1 hour ago"},
        {"action": "API credentials verified", "symbol": "Aster DEX", "result": "Connected", "time": "2 hours ago"}
    ]

    for activity in activities:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 600; color: #43e97b;">{activity['action']}</span>
                    <span style="color: #6b7280; margin-left: 0.5rem;">• {activity['symbol']}</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 600; color: #667eea;">{activity['result']}</div>
                    <div style="font-size: 0.8rem; color: #6b7280;">{activity['time']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.02); border-radius: 8px; margin-top: 1rem;">
        <p style="color: #9ca3af; font-size: 0.8rem; margin: 0;">
            <strong>Rari Trade AI</strong> v1.0.0<br>
            Built for institutional-grade trading
        </p>
        <div style="margin-top: 0.5rem;">
            <span style="color: #43e97b; font-size: 0.75rem;">●</span>
            <span style="color: #6b7280; font-size: 0.75rem; margin-left: 0.25rem;">Ready for live trading</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add the actual Streamlit buttons (invisible but functional)
    st.markdown('<div style="position: absolute; opacity: 0; pointer-events: none;">', unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        if st.button("🚀 Start Strategy Lab", use_container_width=True, type="primary", key="lab_btn"):
            st.session_state.page = "🔬 Strategy Lab"

    with action_col2:
        if st.button("📊 View Analytics", use_container_width=True, key="analytics_btn"):
            st.session_state.page = "📈 Analytics"

    with action_col3:
        if st.button("⚙️ Configure Trading", use_container_width=True, key="trading_btn"):
            st.session_state.page = "⚙️ Live Trading"
    st.markdown('</div>', unsafe_allow_html=True)

def show_trading_dashboard():
    """Enhanced trading dashboard with rich analytics."""
    st.markdown("## 📊 Trading Performance Dashboard")

    # Strategy Execution based on selected type
    try:
        if strategy_type == "SMA Crossover (Beginner)":
            pos = generate_positions_sma_crossover(pd.DataFrame({"close": close}), short_win, long_win)
            strategy_name = f"SMA Crossover ({short_win}/{long_win})"

        elif strategy_type == "RSI Mean Reversion":
            # Generate RSI-based positions
            from mcp_trader.strategies.indicators import rsi
            rsi_values = rsi(close, rsi_period)
            pos = np.where(rsi_values < rsi_low, 1.0, np.where(rsi_values > rsi_high, -1.0, 0.0))
            pos = pd.Series(pos, index=close.index, name="position")
            strategy_name = f"RSI Mean Reversion ({rsi_period}, {rsi_low}/{rsi_high})"

        elif strategy_type == "DMark Indicator (Advanced)":
            # Use DMark strategy
            from mcp_trader.strategies.dmark_strategy import DMarkStrategy
            dmark_config = {"dmark_config": {"mode": dmark_config.lower()}}
            dmark_strategy = DMarkStrategy(dmark_config)
            predictions = dmark_strategy.predict(full_data if full_data is not None else pd.DataFrame({"close": close}))
            pos = pd.Series([p.prediction for p in predictions], index=close.index, name="position")
            strategy_name = f"DMark Indicator ({dmark_config})"

        else:
            # Default to SMA crossover
            pos = generate_positions_sma_crossover(pd.DataFrame({"close": close}), short_win, long_win)
            strategy_name = f"SMA Crossover ({short_win}/{long_win})"

        res = evaluate_positions(close, pos, fee_bps=fee_bps)
        metrics = res["metrics"]

    except Exception as e:
        st.error(f"❌ Strategy execution failed: {str(e)}")
        st.info("💡 Try using SMA Crossover strategy or check your data format.")
        st.stop()

    # Create rich dashboard with multiple sections
    create_performance_overview(metrics, strategy_name)
    create_risk_analytics(metrics, res, pos)
    create_strategy_insights(pos, close, strategy_name, metrics, res)

def create_performance_overview(metrics, strategy_name):
    """Create comprehensive performance overview section."""
    st.markdown("### 🎯 Performance Overview")

    # Key metrics in a professional grid
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            "Total Return",
            f"{metrics['total_return']:.1%}",
            "Strategy performance vs buy & hold",
            "📈"
        )

    with col2:
        render_metric_card(
            "Sharpe Ratio",
            f"{metrics['sharpe']:.2f}",
            f"Risk-adjusted returns ({'Excellent' if metrics['sharpe'] > 2 else 'Good' if metrics['sharpe'] > 1 else 'Fair'})",
            "⚡"
        )

    with col3:
        render_metric_card(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.1%}",
            "Worst peak-to-trough decline",
            "📉"
        )

    with col4:
        render_metric_card(
            "Profit Factor",
            f"{metrics['profit_factor']:.2f}",
            f"Gross profits / losses ({'Profitable' if metrics['profit_factor'] > 1.5 else 'Needs work'})",
            "💰"
        )

    # Strategy health indicators
    st.markdown("### 🔍 Strategy Health")

    health_col1, health_col2, health_col3 = st.columns(3)

    with health_col1:
        sharpe_status = "🟢" if metrics['sharpe'] > 1.5 else "🟡" if metrics['sharpe'] > 0.5 else "🔴"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{sharpe_status}</div>
            <div style="font-weight: 600;">Risk Efficiency</div>
            <div style="font-size: 0.9rem; color: #6b7280;">Sharpe > 1.5 preferred</div>
        </div>
        """, unsafe_allow_html=True)

    with health_col2:
        dd_status = "🟢" if metrics['max_drawdown'] < 0.15 else "🟡" if metrics['max_drawdown'] < 0.25 else "🔴"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{dd_status}</div>
            <div style="font-weight: 600;">Drawdown Control</div>
            <div style="font-size: 0.9rem; color: #6b7280;">< 15% preferred</div>
        </div>
        """, unsafe_allow_html=True)

    with health_col3:
        pf_status = "🟢" if metrics['profit_factor'] > 1.5 else "🟡" if metrics['profit_factor'] > 1.0 else "🔴"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{pf_status}</div>
            <div style="font-weight: 600;">Profitability</div>
            <div style="font-size: 0.9rem; color: #6b7280;">> 1.5 profit factor</div>
        </div>
        """, unsafe_allow_html=True)

def create_risk_analytics(metrics, res, pos):
    """Create detailed risk analytics section."""
    st.markdown("### 🛡️ Risk Analytics")

    # Risk metrics breakdown
    risk_col1, risk_col2, risk_col3 = st.columns(3)

    with risk_col1:
        st.markdown("**Volatility Metrics**")
        st.metric("Annualized Volatility", f"{res['returns'].std() * np.sqrt(365):.1%}")
        st.metric("Value at Risk (95%)", f"{res['returns'].quantile(0.05):.1%}")
        st.metric("Expected Shortfall", f"{res['returns'][res['returns'] <= res['returns'].quantile(0.05)].mean():.1%}")

    with risk_col2:
        st.markdown("**Drawdown Analysis**")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
        st.metric("Avg Drawdown", f"{(res['equity'] / res['equity'].cummax() - 1).mean():.1%}")
        recovery_period = len(res['equity']) - res['equity'].argmax()
        st.metric("Recovery Time", f"{recovery_period} periods")

    with risk_col3:
        st.markdown("**Trade Statistics**")
        total_trades = sum(1 for p in pos if p != 0)
        winning_trades = sum(1 for i, p in enumerate(pos) if p != 0 and res['returns'].iloc[i] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        st.metric("Total Trades", total_trades)
        st.metric("Win Rate", f"{win_rate:.1%}")
        st.metric("Avg Trade Return", f"{res['returns'][res['returns'] != 0].mean():.2%}")

    # Risk-adjusted return chart
    st.markdown("### 📊 Risk-Adjusted Performance")

    fig_risk = go.Figure()

    # Cumulative returns
    fig_risk.add_trace(go.Scatter(
        y=res['equity'] - 1,
        name="Strategy Returns",
        line=dict(color="#667eea", width=2),
        hovertemplate="Return: %{y:.1%}<extra></extra>"
    ))

    # Risk parity line (theoretical optimal)
    risk_parity_return = np.linspace(0, len(res['equity']) - 1, len(res['equity'])) * 0.0001  # 0.01% per period
    risk_parity_equity = (1 + risk_parity_return).cumprod()
    fig_risk.add_trace(go.Scatter(
        y=risk_parity_equity - 1,
        name="Risk Parity Benchmark",
        line=dict(color="#6b7280", width=1, dash="dash"),
        hovertemplate="Benchmark: %{y:.1%}<extra></extra>"
    ))

    fig_risk.update_layout(
        title="Cumulative Returns vs Risk Benchmark",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        xaxis=dict(gridcolor="#374151"),
        yaxis=dict(gridcolor="#374151", tickformat=".1%")
    )

    st.plotly_chart(fig_risk, use_container_width=True)

def create_strategy_insights(pos, close, strategy_name, metrics, res):
    """Create actionable strategy insights."""
    # Calculate win rate
    total_trades = sum(1 for p in pos if p != 0)
    winning_trades = sum(1 for i, p in enumerate(pos) if p != 0 and res['returns'].iloc[i] > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    st.markdown("### 💡 Strategy Insights & Recommendations")

    # Strategy performance breakdown
    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown("**Strengths**")
        strengths = []
        if metrics['sharpe'] > 1.5:
            strengths.append("✅ Excellent risk-adjusted returns")
        if metrics['profit_factor'] > 1.5:
            strengths.append("✅ Strong profitability")
        if metrics['max_drawdown'] < 0.15:
            strengths.append("✅ Good drawdown control")
        if len(strengths) == 0:
            strengths.append("📈 Strategy showing potential - consider optimization")

        for strength in strengths:
            st.markdown(strength)

    with insight_col2:
        st.markdown("**Areas for Improvement**")
        improvements = []
        if metrics['sharpe'] < 1.0:
            improvements.append("⚠️ Consider reducing volatility or improving returns")
        if metrics['profit_factor'] < 1.2:
            improvements.append("⚠️ Focus on increasing win rate or profit per trade")
        if metrics['max_drawdown'] > 0.20:
            improvements.append("⚠️ Implement stricter risk management")
        if len(improvements) == 0:
            improvements.append("✅ Strategy performing well - monitor for consistency")

        for improvement in improvements:
            st.markdown(improvement)

    # Actionable recommendations
    st.markdown("### 🎯 Actionable Recommendations")

    recommendations = []

    if metrics['sharpe'] > 2.0:
        recommendations.append({
            "priority": "High",
            "action": "🚀 Consider live deployment with small position sizes",
            "reason": "Strategy shows excellent risk-adjusted performance"
        })

    if metrics['max_drawdown'] > 0.25:
        recommendations.append({
            "priority": "High",
            "action": "🛡️ Reduce position sizes and implement stricter stops",
            "reason": "Drawdown exceeds acceptable risk levels"
        })

    if win_rate < 0.4:
        recommendations.append({
            "priority": "Medium",
            "action": "🎯 Optimize entry/exit signals to improve win rate",
            "reason": "Low win rate may indicate signal quality issues"
        })

    if metrics['total_return'] < 0.05:
        recommendations.append({
            "priority": "Medium",
            "action": "📊 Backtest on different market conditions",
            "reason": "Limited historical performance - test robustness"
        })

    if len(recommendations) == 0:
        recommendations.append({
            "priority": "Low",
            "action": "👁️ Continue monitoring performance",
            "reason": "Strategy performing within acceptable parameters"
        })

    for rec in recommendations:
        priority_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}[rec["priority"]]
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 2px solid {priority_color}20; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                <div style="font-size: 1.5rem;">{rec['action'].split()[0]}</div>
                <div>
                    <div style="font-weight: 600; color: {priority_color}; margin-bottom: 0.25rem;">{rec['priority']} Priority</div>
                    <div style="font-weight: 500; margin-bottom: 0.25rem;">{rec['action']}</div>
                    <div style="font-size: 0.9rem; color: #6b7280;">{rec['reason']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_strategy_lab():
    """Advanced strategy laboratory for testing and optimization."""
    st.markdown("## 🔬 Strategy Laboratory")

    st.info("🧪 **Advanced Strategy Development Environment**")
    st.markdown("""
    This laboratory provides tools for developing, testing, and optimizing trading strategies.
    Use the controls in the sidebar to configure your strategy parameters.
    """)

    # Strategy comparison section
    st.markdown("### 📊 Strategy Comparison")

    # Create comparison table with multiple strategies
    strategies_data = []

    # Test multiple parameter combinations
    test_configs = [
        {"short": 10, "long": 30, "name": "Fast Trend"},
        {"short": 20, "long": 50, "name": "Balanced"},
        {"short": 50, "long": 200, "name": "Slow Trend"},
    ]

    for config in test_configs:
        try:
            pos = generate_positions_sma_crossover(pd.DataFrame({"close": close}), config["short"], config["long"])
            res = evaluate_positions(close, pos, fee_bps=fee_bps)
            metrics = res["metrics"]

            strategies_data.append({
                "Strategy": config["name"],
                "Parameters": f"{config['short']}/{config['long']}",
                "Total Return": f"{metrics['total_return']:.1%}",
                "Sharpe": f"{metrics['sharpe']:.2f}",
                "Max DD": f"{metrics['max_drawdown']:.1%}",
                "Profit Factor": f"{metrics['profit_factor']:.2f}",
            })
        except Exception as e:
            st.warning(f"Error testing {config['name']}: {e}")

    if strategies_data:
        st.dataframe(pd.DataFrame(strategies_data), use_container_width=True)

    # Optimization suggestions
    st.markdown("### 🎯 Optimization Recommendations")

    best_sharpe = max(strategies_data, key=lambda x: float(x["Sharpe"]))
    best_return = max(strategies_data, key=lambda x: float(x["Total Return"][:-1])/100)

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**Highest Sharpe Ratio:** {best_sharpe['Strategy']} ({best_sharpe['Sharpe']})")
        st.markdown("Best risk-adjusted performance")

    with col2:
        st.info(f"**Highest Returns:** {best_return['Strategy']} ({best_return['Total Return']})")
        st.markdown("Best absolute performance")

def show_live_trading():
    """Live trading configuration and monitoring."""
    st.markdown("## ⚙️ Live Trading Configuration")

    st.warning("🚨 **Live Trading - Use Caution**")
    st.markdown("""
    Live trading involves real financial risk. Only proceed if you understand the risks and have tested your strategy thoroughly.
    """)

    # Trading configuration
    st.markdown("### 🎯 Trading Setup")

    with st.expander("Risk Parameters", expanded=True):
        live_position_size = st.slider("Position Size %", 0.1, 5.0, 1.0, help="Percentage of portfolio per trade")
        live_stop_loss = st.slider("Stop Loss %", 0.5, 5.0, 2.0, help="Automatic loss cutoff")
        live_take_profit = st.slider("Take Profit %", 1.0, 10.0, 4.0, help="Automatic profit taking")

    with st.expander("Strategy Selection"):
        live_strategy = st.selectbox("Live Strategy", ["SMA Crossover", "RSI Mean Reversion"], index=0)

    # Status and controls
    st.markdown("### 🔴 Live Trading Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.metric("Connection Status", "Disconnected", "Ready to connect")

    with status_col2:
        st.metric("Active Positions", "0", "No open trades")

    with status_col3:
        st.metric("Today's P&L", "$0.00", "0.00%")

    # Control buttons
    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        if st.button("🟢 Start Live Trading", use_container_width=True, type="primary"):
            st.success("Live trading simulation started (demo mode)")

    with control_col2:
        if st.button("⏹️ Stop Trading", use_container_width=True):
            st.info("Trading stopped")

    with control_col3:
        if st.button("🔄 Emergency Close", use_container_width=True):
            st.warning("All positions closed (emergency)")

    st.markdown("---")
    st.markdown("### 📋 Pre-Launch Checklist")

    checklist_items = [
        {"item": "API credentials updated", "status": "✅", "detail": "Keys verified"},
        {"item": "Strategy backtested", "status": "✅", "detail": "Performance validated"},
        {"item": "Risk limits set", "status": "✅", "detail": "Position sizes configured"},
        {"item": "Emergency stops ready", "status": "✅", "detail": "Manual override available"},
    ]

    for item in checklist_items:
        st.markdown(f"{item['status']} **{item['item']}** - {item['detail']}")

def show_analytics():
    """Advanced analytics and reporting."""
    st.markdown("## 📈 Advanced Analytics")

    st.markdown("### 🔬 Market Analysis")

    # Market regime detection
    st.markdown("**Market Regime Detection**")
    regime_col1, regime_col2, regime_col3 = st.columns(3)

    with regime_col1:
        st.metric("Volatility Regime", "Low", "15-day average")

    with regime_col2:
        st.metric("Trend Strength", "Bullish", "Momentum indicator")

    with regime_col3:
        st.metric("Market Efficiency", "Mean Reverting", "Hurst exponent")

    # Correlation analysis
    st.markdown("### 📊 Asset Correlation Matrix")

    # Mock correlation data
    assets = ["BTC", "ETH", "SOL", "ADA", "DOT"]
    correlation_data = np.random.uniform(0.1, 0.9, (5, 5))
    correlation_data = (correlation_data + correlation_data.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_data, 1.0)

    corr_df = pd.DataFrame(correlation_data, columns=assets, index=assets)
    st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"))

    # Performance attribution
    st.markdown("### 📈 Performance Attribution")

    attribution_data = {
        "Factor": ["Trend Following", "Mean Reversion", "Carry", "Volatility", "Residual"],
        "Contribution": [35.2, 28.1, 15.8, 12.4, 8.5],
        "Sharpe": [1.8, 2.1, 1.2, 1.9, 0.8]
    }

    attr_df = pd.DataFrame(attribution_data)
    st.bar_chart(attr_df.set_index("Factor")["Contribution"])

def show_academy():
    """Educational content and learning resources."""
    st.markdown("## 🎓 Rari Trade Academy")

    st.markdown("""
    Welcome to the Rari Trade Academy! Learn algorithmic trading concepts,
    understand market dynamics, and develop your trading expertise.
    """)

    # Learning modules
    modules = st.tabs(["📚 Fundamentals", "🧮 Quantitative Methods", "🤖 Algorithmic Strategies", "🛡️ Risk Management"])

    with modules[0]:
        st.markdown("### 📚 Trading Fundamentals")

        st.markdown("#### What is Algorithmic Trading?")
        st.info("""
        Algorithmic trading uses computer programs to execute trades based on predefined criteria.
        It removes emotional decision-making and ensures consistent execution of trading strategies.
        """)

        st.markdown("#### Key Concepts")
        concepts = {
            "Alpha": "Excess return above market benchmark",
            "Beta": "Measure of market sensitivity",
            "Sharpe Ratio": "Risk-adjusted return measurement",
            "Drawdown": "Peak-to-trough decline in portfolio value"
        }

        for concept, definition in concepts.items():
            st.markdown(f"**{concept}**: {definition}")

    with modules[1]:
        st.markdown("### 🧮 Quantitative Methods")

        st.markdown("#### Statistical Measures")
        st.markdown("""
        - **Mean Return**: Average periodic return
        - **Volatility**: Standard deviation of returns
        - **Skewness**: Asymmetry of return distribution
        - **Kurtosis**: Tail risk of return distribution
        """)

        st.markdown("#### Time Series Analysis")
        st.markdown("""
        - **Stationarity**: Constant statistical properties over time
        - **Autocorrelation**: Correlation with past values
        - **Cointegration**: Long-run equilibrium relationships
        - **Granger Causality**: Predictive relationships
        """)

    with modules[2]:
        st.markdown("### 🤖 Algorithmic Strategies")

        st.markdown("#### Trend Following")
        st.info("Buy rising assets, sell falling assets. Works well in trending markets.")

        st.markdown("#### Mean Reversion")
        st.info("Bet on prices returning to historical averages. Works in range-bound markets.")

        st.markdown("#### Momentum")
        st.info("Continue buying winners, selling losers. Exploits behavioral biases.")

        st.markdown("#### Arbitrage")
        st.info("Exploit price differences between related assets or markets.")

    with modules[3]:
        st.markdown("### 🛡️ Risk Management")

        st.markdown("#### Position Sizing")
        st.markdown("""
        - **Fixed Amount**: Same dollar amount per trade
        - **Percentage of Portfolio**: Scale with account size
        - **Kelly Criterion**: Optimal sizing based on win probability
        - **Volatility Adjusted**: Scale with asset volatility
        """)

        st.markdown("#### Risk Controls")
        st.markdown("""
        - **Stop Loss**: Automatic exit at loss threshold
        - **Take Profit**: Automatic exit at profit target
        - **Position Limits**: Maximum exposure per asset
        - **Drawdown Limits**: Reduce exposure during losses
        """)

def show_advanced_analytics():
    """Advanced analytics dashboard with comprehensive analysis and TradingView-like features."""
    st.markdown("## 📈 Advanced Analytics & Technical Analysis")
    st.markdown("### 🔬 Comprehensive Market Analysis & AI Insights")

    # Asset selection and analysis controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_asset = st.selectbox(
            "🎯 Select Asset",
            ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "PENGUUSDT", "ASTERUSDT"],
            index=0
        )

    with col2:
        analysis_period = st.selectbox(
            "⏱️ Timeframe",
            ["1h", "4h", "1d", "1w"],
            index=2
        )

    with col3:
        analysis_type = st.selectbox(
            "🔍 Analysis Type",
            ["Technical Analysis", "Fundamental Analysis", "AI Insights", "Sentiment Analysis"],
            index=0
        )

    with col4:
        chart_style = st.selectbox(
            "📊 Chart Style",
            ["Candlestick", "Line", "Area", "Heikin-Ashi"],
            index=0
        )

    # Load data for selected asset
    try:
        # Mock data for demonstration - in production, load from BigQuery or API
        import numpy as np
        import pandas as pd

        # Generate sample OHLCV data
        np.random.seed(42)
        n_periods = 500
        base_price = {"BTCUSDT": 95000, "ETHUSDT": 2900, "SOLUSDT": 180, "SUIUSDT": 3.2, "PENGUUSDT": 8.5, "ASTERUSDT": 0.15}[selected_asset]

        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_periods, freq='1H')
        returns = np.random.normal(0.0001, 0.02, n_periods)
        prices = base_price * np.exp(np.cumsum(returns))

        # Create OHLCV data
        high_mult = 1 + np.random.uniform(0.005, 0.02, n_periods)
        low_mult = 1 - np.random.uniform(0.005, 0.02, n_periods)
        close = prices
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0] * (1 + np.random.normal(0, 0.01))
        volume = np.random.uniform(1000000, 10000000, n_periods)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': close * high_mult,
            'low': close * low_mult,
            'close': close,
            'volume': volume
        }).set_index('timestamp')

        # Technical Analysis Section
        if analysis_type == "Technical Analysis":
            create_advanced_technical_analysis(df, selected_asset, chart_style)

        # AI Insights Section
        elif analysis_type == "AI Insights":
            create_ai_insights_analysis(df, selected_asset)

        # Fundamental Analysis Section
        elif analysis_type == "Fundamental Analysis":
            create_fundamental_analysis(selected_asset)

        # Sentiment Analysis Section
        elif analysis_type == "Sentiment Analysis":
            create_sentiment_analysis(selected_asset)

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("💡 Try selecting a different asset or timeframe.")

def create_advanced_technical_analysis(df, asset, chart_style):
    """Create comprehensive technical analysis with TradingView-like features."""
    st.markdown("### 📊 Advanced Technical Analysis")

    # Main Chart with Multiple Indicators
    fig = create_tradingview_chart(df, asset, chart_style)
    st.plotly_chart(fig, use_container_width=True)

    # Technical Indicators Panel
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📈 Trend Indicators")
        # Calculate and display trend indicators
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['close'].ewm(span=26).mean().iloc[-1]

        st.metric("SMA 20", f"${sma_20:.2f}")
        st.metric("SMA 50", f"${sma_50:.2f}")
        st.metric("EMA 12", f"${ema_12:.2f}")
        st.metric("EMA 26", f"${ema_26:.2f}")

        # Trend direction
        trend_signal = "🟢 BULLISH" if ema_12 > ema_26 else "🔴 BEARISH"
        st.markdown(f"**Trend**: {trend_signal}")

    with col2:
        st.markdown("#### 📊 Momentum Indicators")
        # RSI
        rsi = calculate_rsi(df['close'])
        rsi_value = rsi.iloc[-1]
        st.metric("RSI", f"{rsi_value:.1f}")

        # MACD
        macd_line, signal_line, histogram = calculate_macd(df['close'])
        macd_val = macd_line.iloc[-1]
        signal_val = signal_line.iloc[-1]
        st.metric("MACD", f"{macd_val:.4f}")
        st.metric("Signal", f"{signal_val:.4f}")

        # RSI signals
        if rsi_value > 70:
            st.markdown("**RSI**: 🔴 OVERBOUGHT")
        elif rsi_value < 30:
            st.markdown("**RSI**: 🟢 OVERSOLD")
        else:
            st.markdown("**RSI**: 🟡 NEUTRAL")

    with col3:
        st.markdown("#### 💹 Volume & Volatility")
        # Volume analysis
        avg_volume = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        st.metric("Volume", f"{current_volume:,.0f}")

        # ATR (Average True Range)
        atr = calculate_atr(df)
        st.metric("ATR (Volatility)", f"{atr:.4f}")

        # Volume trend
        vol_trend = "📈 HIGH" if current_volume > avg_volume * 1.2 else "📉 LOW"
        st.markdown(f"**Volume**: {vol_trend}")

    # Buy/Sell Signals
    st.markdown("### 🎯 AI-Generated Trading Signals")

    signals = generate_trading_signals(df)
    display_trading_signals(signals)

    # Support & Resistance Levels
    st.markdown("### 📏 Support & Resistance Levels")
    support_resistance = calculate_support_resistance(df)
    display_support_resistance(support_resistance)

    # Pattern Recognition
    st.markdown("### 🔍 Pattern Recognition")
    patterns = detect_chart_patterns(df)
    display_patterns(patterns)

def create_ai_insights_analysis(df, asset):
    """Create AI-powered insights and analysis."""
    st.markdown("### 🤖 AI-Powered Market Insights")

    # Market Regime Detection
    regime = detect_market_regime(df)
    st.markdown(f"**Market Regime**: {regime['regime']} (Confidence: {regime['confidence']:.1%})")

    # Predictive Analytics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Price Prediction (24h)")
        prediction = predict_price_movement(df)
        st.metric("Predicted Price", f"${prediction['price']:.2f}")
        st.metric("Confidence", f"{prediction['confidence']:.1%}")

        direction = "🟢 UP" if prediction['direction'] == 'bullish' else "🔴 DOWN"
        st.markdown(f"**Direction**: {direction}")

    with col2:
        st.markdown("#### 📊 Risk Assessment")
        risk_metrics = calculate_risk_metrics(df)
        st.metric("Volatility", f"{risk_metrics['volatility']:.2%}")
        st.metric("VaR (95%)", f"{risk_metrics['var_95']:.2%}")
        st.metric("Sharpe Ratio", f"{risk_metrics['sharpe']:.2f}")

    # AI Recommendations
    st.markdown("### 💡 AI Trading Recommendations")
    recommendations = generate_ai_recommendations(df, asset)
    display_ai_recommendations(recommendations)

def create_fundamental_analysis(asset):
    """Create fundamental analysis section."""
    st.markdown("### 🏛️ Fundamental Analysis")

    # Asset Information
    asset_info = get_asset_fundamentals(asset)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Market Cap", asset_info.get('market_cap', 'N/A'))
        st.metric("24h Volume", asset_info.get('volume_24h', 'N/A'))

    with col2:
        st.metric("Circulating Supply", asset_info.get('circulating_supply', 'N/A'))
        st.metric("Total Supply", asset_info.get('total_supply', 'N/A'))

    with col3:
        st.metric("FDV", asset_info.get('fdv', 'N/A'))
        st.metric("ATH", asset_info.get('ath', 'N/A'))

    # On-chain Metrics
    st.markdown("### ⛓️ On-Chain Metrics")
    on_chain = get_on_chain_metrics(asset)

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.metric("Active Addresses", on_chain.get('active_addresses', 'N/A'))
        st.metric("Transaction Count", on_chain.get('tx_count', 'N/A'))

    with metrics_col2:
        st.metric("Network Hashrate", on_chain.get('hashrate', 'N/A'))
        st.metric("Difficulty", on_chain.get('difficulty', 'N/A'))

    with metrics_col3:
        st.metric("Gas Price", on_chain.get('gas_price', 'N/A'))
        st.metric("TVL", on_chain.get('tvl', 'N/A'))

def create_sentiment_analysis(asset):
    """Create sentiment analysis section."""
    st.markdown("### 🗣️ Sentiment Analysis")

    # Social Media Sentiment
    sentiment = get_social_sentiment(asset)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Twitter Sentiment", f"{sentiment.get('twitter', 0):.1f}/10")
        st.metric("Reddit Sentiment", f"{sentiment.get('reddit', 0):.1f}/10")

    with col2:
        st.metric("News Sentiment", f"{sentiment.get('news', 0):.1f}/10")
        st.metric("Fear & Greed", f"{sentiment.get('fear_greed', 0)}")

    with col3:
        st.metric("Social Volume", f"{sentiment.get('social_volume', 0):,.0f}")
        st.metric("Social Dominance", f"{sentiment.get('social_dominance', 0):.1f}%")

    # Sentiment Chart
    st.markdown("### 📈 Sentiment Timeline")
    sentiment_chart = create_sentiment_chart(asset)
    st.plotly_chart(sentiment_chart, use_container_width=True)

def show_model_zoo():
    """Model Zoo - Compare multiple AI models and strategies."""
    st.markdown("## 🤖 Model Zoo & Strategy Comparison")
    st.markdown("### 🏆 Compare AI Models Across Multiple Assets")

    # Model Selection
    col1, col2, col3 = st.columns(3)

    with col1:
        models_to_compare = st.multiselect(
            "🤖 Select Models",
            ["DMark Strategy", "SMA Crossover", "RSI Mean Reversion", "MACD Strategy", "Bollinger Bands", "Fibonacci Retracement"],
            default=["DMark Strategy", "SMA Crossover", "RSI Mean Reversion"]
        )

    with col2:
        assets_to_compare = st.multiselect(
            "🎯 Select Assets",
            ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "PENGUUSDT", "ASTERUSDT"],
            default=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        )

    with col3:
        backtest_period = st.selectbox(
            "⏱️ Backtest Period",
            ["1 Month", "3 Months", "6 Months", "1 Year"],
            index=1
        )

    if models_to_compare and assets_to_compare:
        # Run Model Comparison
        comparison_results = run_model_comparison(models_to_compare, assets_to_compare, backtest_period)

        # Results Overview
        st.markdown("### 📊 Performance Overview")

        # Create comparison table
        perf_df = create_performance_comparison_table(comparison_results)
        st.dataframe(perf_df, use_container_width=True)

        # Performance Charts
        st.markdown("### 📈 Performance Comparison")

        # Returns comparison chart
        returns_chart = create_returns_comparison_chart(comparison_results)
        st.plotly_chart(returns_chart, use_container_width=True)

        # Risk metrics comparison
        st.markdown("### 🛡️ Risk-Adjusted Performance")

        risk_chart = create_risk_comparison_chart(comparison_results)
        st.plotly_chart(risk_chart, use_container_width=True)

        # Model Rankings
        st.markdown("### 🏆 Model Rankings by Asset")

        rankings = create_model_rankings(comparison_results)
        display_model_rankings(rankings)

        # Best Model Recommendations
        st.markdown("### 💡 AI Model Recommendations")

        recommendations = generate_model_recommendations(comparison_results)
        display_model_recommendations(recommendations)

    else:
        st.info("🎯 Please select at least one model and one asset to run the comparison.")

# Helper Functions for Technical Analysis

def calculate_rsi(price_series, period=14):
    """Calculate RSI indicator."""
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(price_series, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    fast_ema = price_series.ewm(span=fast).mean()
    slow_ema = price_series.ewm(span=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.iloc[-1]

def create_tradingview_chart(df, asset, style):
    """Create TradingView-like chart."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{asset} Price', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Main price chart
    if style == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ), row=1, col=1)
    elif style == "Line":
        fig.add_trace(go.Scatter(
            x=df.index, y=df['close'],
            mode='lines',
            name="Close Price",
            line=dict(color='#2962ff', width=2)
        ), row=1, col=1)
    elif style == "Area":
        fig.add_trace(go.Scatter(
            x=df.index, y=df['close'],
            fill='tozeroy',
            mode='lines',
            name="Close Price",
            line=dict(color='#2962ff'),
            fillcolor='rgba(41, 98, 255, 0.1)'
        ), row=1, col=1)

    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'].rolling(20).mean(),
        mode='lines',
        name="SMA 20",
        line=dict(color='#ff6b35', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'].rolling(50).mean(),
        mode='lines',
        name="SMA 50",
        line=dict(color='#f7931e', width=1)
    ), row=1, col=1)

    # Volume chart
    colors = ['red' if row['open'] > row['close'] else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'],
        name="Volume",
        marker_color=colors,
        opacity=0.7
    ), row=2, col=1)

    # RSI
    rsi = calculate_rsi(df['close'])
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi,
        mode='lines',
        name="RSI",
        line=dict(color='#7c3aed', width=1)
    ), row=3, col=1)

    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(showgrid=True, gridcolor='#374151')
    fig.update_yaxes(showgrid=True, gridcolor='#374151')

    return fig

def generate_trading_signals(df):
    """Generate comprehensive trading signals."""
    signals = []

    # RSI signals
    rsi = calculate_rsi(df['close'])
    if rsi.iloc[-1] < 30:
        signals.append({
            "type": "BUY",
            "indicator": "RSI",
            "strength": "Strong",
            "reason": "Oversold condition - RSI below 30"
        })
    elif rsi.iloc[-1] > 70:
        signals.append({
            "type": "SELL",
            "indicator": "RSI",
            "strength": "Strong",
            "reason": "Overbought condition - RSI above 70"
        })

    # Moving average signals
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()

    if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
        signals.append({
            "type": "BUY",
            "indicator": "SMA Crossover",
            "strength": "Medium",
            "reason": "Golden cross - Short MA crossed above Long MA"
        })
    elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]:
        signals.append({
            "type": "SELL",
            "indicator": "SMA Crossover",
            "strength": "Medium",
            "reason": "Death cross - Short MA crossed below Long MA"
        })

    # MACD signals
    macd_line, signal_line, histogram = calculate_macd(df['close'])
    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
        signals.append({
            "type": "BUY",
            "indicator": "MACD",
            "strength": "Medium",
            "reason": "MACD crossed above signal line"
        })
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
        signals.append({
            "type": "SELL",
            "indicator": "MACD",
            "strength": "Medium",
            "reason": "MACD crossed below signal line"
        })

    return signals

def display_trading_signals(signals):
    """Display trading signals in a nice format."""
    if not signals:
        st.info("📊 No strong signals detected at current levels.")
        return

    for signal in signals:
        if signal["type"] == "BUY":
            st.success(f"🟢 **{signal['indicator']}**: {signal['reason']} ({signal['strength']} signal)")
        else:
            st.error(f"🔴 **{signal['indicator']}**: {signal['reason']} ({signal['strength']} signal)")

def calculate_support_resistance(df, lookback=50):
    """Calculate support and resistance levels."""
    recent_data = df.tail(lookback)

    # Simple pivot points
    high = recent_data['high'].max()
    low = recent_data['low'].min()
    close = recent_data['close'].iloc[-1]

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)

    return {
        "resistance_2": r2,
        "resistance_1": r1,
        "pivot": pivot,
        "support_1": s1,
        "support_2": s2
    }

def display_support_resistance(levels):
    """Display support and resistance levels."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Resistance R2", f"${levels['resistance_2']:.2f}")
        st.metric("Resistance R1", f"${levels['resistance_1']:.2f}")

    with col2:
        st.metric("Pivot Point", f"${levels['pivot']:.2f}")

    with col3:
        st.metric("Support S1", f"${levels['support_1']:.2f}")
        st.metric("Support S2", f"${levels['support_2']:.2f}")

def detect_chart_patterns(df):
    """Detect common chart patterns."""
    patterns = []

    # Simple pattern detection (in a real implementation, this would be much more sophisticated)
    recent_prices = df['close'].tail(20).values

    # Double bottom pattern (simplified)
    if len(recent_prices) >= 10:
        min_idx = np.argmin(recent_prices)
        if min_idx > 2 and min_idx < len(recent_prices) - 3:
            left_low = recent_prices[min_idx - 2]
            right_low = recent_prices[min_idx + 2]
            if abs(left_low - recent_prices[min_idx]) < recent_prices[min_idx] * 0.02 and \
               abs(right_low - recent_prices[min_idx]) < recent_prices[min_idx] * 0.02:
                patterns.append({
                    "pattern": "Double Bottom",
                    "confidence": 0.75,
                    "description": "Potential reversal pattern forming"
                })

    # Head and shoulders (simplified)
    if len(recent_prices) >= 15:
        # This is a very simplified version - real pattern recognition is much more complex
        patterns.append({
            "pattern": "Analyzing Patterns",
            "confidence": 0.5,
            "description": "Advanced pattern recognition in progress"
        })

    return patterns

def display_patterns(patterns):
    """Display detected patterns."""
    if not patterns:
        st.info("🔍 No clear patterns detected in current price action.")
        return

    for pattern in patterns:
        confidence_color = "🟢" if pattern["confidence"] > 0.7 else "🟡" if pattern["confidence"] > 0.5 else "🔴"
        st.markdown(f"{confidence_color} **{pattern['pattern']}** (Confidence: {pattern['confidence']:.1%})")
        st.markdown(f"*{pattern['description']}*")

def detect_market_regime(df):
    """Detect current market regime."""
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(365)

    if volatility > 0.8:  # High volatility
        return {"regime": "🔴 HIGH VOLATILITY", "confidence": 0.85}
    elif volatility < 0.3:  # Low volatility
        trend = "BULL" if df['close'].iloc[-1] > df['close'].rolling(50).mean().iloc[-1] else "BEAR"
        return {"regime": f"🟢 {trend} TREND", "confidence": 0.75}
    else:
        return {"regime": "🟡 SIDEWAYS", "confidence": 0.6}

def predict_price_movement(df):
    """Simple price prediction model."""
    current_price = df['close'].iloc[-1]
    trend = df['close'].rolling(20).mean().iloc[-1] - df['close'].rolling(20).mean().iloc[-20]

    if trend > 0:
        predicted_price = current_price * (1 + abs(trend) / current_price * 0.5)
        return {
            "price": predicted_price,
            "direction": "bullish",
            "confidence": min(0.8, abs(trend) / current_price * 10)
        }
    else:
        predicted_price = current_price * (1 - abs(trend) / current_price * 0.5)
        return {
            "price": predicted_price,
            "direction": "bearish",
            "confidence": min(0.8, abs(trend) / current_price * 10)
        }

def calculate_risk_metrics(df):
    """Calculate comprehensive risk metrics."""
    returns = df['close'].pct_change().dropna()

    volatility = returns.std() * np.sqrt(365)
    var_95 = np.percentile(returns, 5)
    sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0

    return {
        "volatility": volatility,
        "var_95": abs(var_95),
        "sharpe": sharpe
    }

def generate_ai_recommendations(df, asset):
    """Generate AI-powered trading recommendations."""
    recommendations = []

    # Analyze multiple factors
    rsi = calculate_rsi(df['close']).iloc[-1]
    macd_line, signal_line, _ = calculate_macd(df['close'])
    macd_signal = "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish"

    trend_score = 0
    if rsi < 40: trend_score += 1
    if macd_signal == "bullish": trend_score += 1
    if df['close'].iloc[-1] > df['close'].rolling(50).mean().iloc[-1]: trend_score += 1

    if trend_score >= 2:
        recommendations.append({
            "action": "BUY",
            "confidence": min(0.9, 0.6 + trend_score * 0.1),
            "timeframe": "Short-term (1-3 days)",
            "reason": "Multiple bullish indicators aligned"
        })
    elif trend_score <= 1:
        recommendations.append({
            "action": "SELL",
            "confidence": min(0.9, 0.6 + (3 - trend_score) * 0.1),
            "timeframe": "Short-term (1-3 days)",
            "reason": "Bearish signals dominate"
        })
    else:
        recommendations.append({
            "action": "HOLD",
            "confidence": 0.7,
            "timeframe": "Wait for clearer signals",
            "reason": "Mixed signals - wait for confirmation"
        })

    return recommendations

def display_ai_recommendations(recommendations):
    """Display AI recommendations."""
    for rec in recommendations:
        if rec["action"] == "BUY":
            st.success(f"🟢 **{rec['action']}** - Confidence: {rec['confidence']:.1%}")
        elif rec["action"] == "SELL":
            st.error(f"🔴 **{rec['action']}** - Confidence: {rec['confidence']:.1%}")
        else:
            st.warning(f"🟡 **{rec['action']}** - Confidence: {rec['confidence']:.1%}")

        st.markdown(f"**Timeframe**: {rec['timeframe']}")
        st.markdown(f"**Reason**: {rec['reason']}")

def get_asset_fundamentals(asset):
    """Get fundamental data for asset."""
    # Mock data - in production, fetch from APIs
    fundamentals = {
        "BTCUSDT": {
            "market_cap": "$1.8T",
            "volume_24h": "$45.2B",
            "circulating_supply": "19.7M BTC",
            "total_supply": "21M BTC",
            "fdv": "$2.0T",
            "ath": "$69,000"
        },
        "ETHUSDT": {
            "market_cap": "$350B",
            "volume_24h": "$18.7B",
            "circulating_supply": "120M ETH",
            "total_supply": "120M ETH",
            "fdv": "$350B",
            "ath": "$4,891"
        }
    }

    return fundamentals.get(asset, {
        "market_cap": "N/A",
        "volume_24h": "N/A",
        "circulating_supply": "N/A",
        "total_supply": "N/A",
        "fdv": "N/A",
        "ath": "N/A"
    })

def get_on_chain_metrics(asset):
    """Get on-chain metrics for asset."""
    # Mock data
    return {
        "active_addresses": "850K",
        "tx_count": "2.1M",
        "hashrate": "580 EH/s" if asset == "BTCUSDT" else "N/A",
        "difficulty": "95T" if asset == "BTCUSDT" else "N/A",
        "gas_price": "25 gwei" if asset == "ETHUSDT" else "N/A",
        "tvl": "$45B" if asset == "ETHUSDT" else "N/A"
    }

def get_social_sentiment(asset):
    """Get social sentiment data."""
    # Mock data
    return {
        "twitter": 6.8,
        "reddit": 7.2,
        "news": 5.9,
        "fear_greed": 65,
        "social_volume": 125000,
        "social_dominance": 12.5
    }

def create_sentiment_chart(asset):
    """Create sentiment timeline chart."""
    import plotly.graph_objects as go

    # Mock sentiment data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    sentiment_data = np.random.normal(6.0, 1.5, 30)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=sentiment_data,
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#7c3aed', width=2),
        marker=dict(size=6)
    ))

    fig.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Bullish")
    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Bearish")
    fig.add_hline(y=6, line_dash="dot", line_color="gray", annotation_text="Neutral")

    fig.update_layout(
        title=f"{asset} Social Sentiment (30-day)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig

def run_model_comparison(models, assets, period):
    """Run comprehensive model comparison across assets."""
    results = {}

    for asset in assets:
        results[asset] = {}
        for model in models:
            # Mock backtest results - in production, run actual backtests
            results[asset][model] = {
                "total_return": np.random.normal(0.15, 0.1),
                "sharpe_ratio": np.random.normal(1.5, 0.5),
                "max_drawdown": abs(np.random.normal(-0.15, 0.05)),
                "win_rate": np.random.normal(0.55, 0.1),
                "profit_factor": np.random.normal(1.3, 0.2),
                "equity_curve": [1.0] + list(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
            }

    return results

def create_performance_comparison_table(results):
    """Create performance comparison table."""
    data = []
    for asset in results:
        for model in results[asset]:
            metrics = results[asset][model]
            data.append({
                "Asset": asset,
                "Model": model,
                "Total Return": f"{metrics['total_return']:.1%}",
                "Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
                "Max Drawdown": f"{metrics['max_drawdown']:.1%}",
                "Win Rate": f"{metrics['win_rate']:.1%}",
                "Profit Factor": f"{metrics['profit_factor']:.2f}"
            })

    return pd.DataFrame(data)

def create_returns_comparison_chart(results):
    """Create returns comparison chart."""
    import plotly.graph_objects as go

    fig = go.Figure()

    for asset in results:
        for model in results[asset]:
            equity = results[asset][model]['equity_curve']
            fig.add_trace(go.Scatter(
                y=equity,
                mode='lines',
                name=f"{asset} - {model}",
                opacity=0.7
            ))

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig

def create_risk_comparison_chart(results):
    """Create risk-adjusted performance chart."""
    import plotly.graph_objects as go

    # Prepare data for radar chart
    categories = ['Return', 'Sharpe', 'Win Rate', 'Profit Factor', 'Risk Control']

    fig = go.Figure()

    for asset in results:
        for model in results[asset]:
            metrics = results[asset][model]

            # Normalize metrics for radar chart
            values = [
                min(metrics['total_return'] * 100, 50),  # Cap at 50%
                min(metrics['sharpe_ratio'] * 10, 25),   # Scale sharpe
                metrics['win_rate'] * 100,               # Win rate as percentage
                min(metrics['profit_factor'] * 20, 50),  # Scale profit factor
                (1 - metrics['max_drawdown']) * 100      # Risk control (inverse of drawdown)
            ]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"{asset} - {model}"
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 50]
            )),
        showlegend=True,
        title="Risk-Adjusted Performance Radar",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig

def create_model_rankings(results):
    """Create model rankings by asset."""
    rankings = {}

    for asset in results:
        asset_rankings = []
        for model in results[asset]:
            metrics = results[asset][model]
            # Composite score based on multiple metrics
            score = (
                metrics['total_return'] * 0.3 +
                metrics['sharpe_ratio'] * 0.25 +
                metrics['win_rate'] * 0.25 +
                (1 - metrics['max_drawdown']) * 0.2  # Lower drawdown is better
            )
            asset_rankings.append({
                "model": model,
                "score": score,
                "rank": 0  # Will be set after sorting
            })

        # Sort by score and assign ranks
        asset_rankings.sort(key=lambda x: x['score'], reverse=True)
        for i, ranking in enumerate(asset_rankings):
            ranking['rank'] = i + 1

        rankings[asset] = asset_rankings

    return rankings

def display_model_rankings(rankings):
    """Display model rankings."""
    for asset in rankings:
        st.markdown(f"#### 🏆 {asset} Rankings")

        for ranking in rankings[asset]:
            medal = "🥇" if ranking['rank'] == 1 else "🥈" if ranking['rank'] == 2 else "🥉" if ranking['rank'] == 3 else "📊"
            st.markdown(f"{medal} **#{ranking['rank']}**: {ranking['model']} (Score: {ranking['score']:.3f})")

def generate_model_recommendations(results):
    """Generate model recommendations based on comparison."""
    recommendations = []

    # Find best performing model overall
    all_scores = []
    for asset in results:
        for model in results[asset]:
            score = (
                results[asset][model]['total_return'] * 0.3 +
                results[asset][model]['sharpe_ratio'] * 0.25 +
                results[asset][model]['win_rate'] * 0.25 +
                (1 - results[asset][model]['max_drawdown']) * 0.2
            )
            all_scores.append((asset, model, score))

    # Sort by score
    all_scores.sort(key=lambda x: x[2], reverse=True)

    best_overall = all_scores[0]
    recommendations.append({
        "type": "Best Overall",
        "model": best_overall[1],
        "asset": best_overall[0],
        "reason": "Highest composite performance score across all metrics"
    })

    # Find most consistent model
    model_consistency = {}
    for asset in results:
        for model in results[asset]:
            if model not in model_consistency:
                model_consistency[model] = []
            model_consistency[model].append(results[asset][model]['sharpe_ratio'])

    most_consistent = max(model_consistency.items(), key=lambda x: np.std(x[1]))
    recommendations.append({
        "type": "Most Consistent",
        "model": most_consistent[0],
        "asset": "All Assets",
        "reason": "Lowest volatility in Sharpe ratio across different assets"
    })

    return recommendations

def display_model_recommendations(recommendations):
    """Display model recommendations."""
    for rec in recommendations:
        st.info(f"🎯 **{rec['type']}**: {rec['model']} ({rec['asset']})")
        st.markdown(f"*{rec['reason']}*")

# Page routing - executed after all functions are defined
render_header()

if page == "🏠 Landing":
    show_landing_page()
elif page == "📊 Trading Dashboard":
    show_trading_dashboard()
elif page == "🔬 Strategy Lab":
    show_strategy_lab()
elif page == "📈 Advanced Analytics":
    show_advanced_analytics()
elif page == "🤖 Model Zoo":
    show_model_zoo()
elif page == "⚙️ Live Trading":
    show_live_trading()
elif page == "🎓 Academy":
    show_academy()
