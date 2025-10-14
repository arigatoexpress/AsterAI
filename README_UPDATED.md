# AIAster - Intelligent Multi-Chain Trading Protocol

## 🚀 Overview

AIAster is a comprehensive, AI-powered trading system designed for Aster DEX perpetual futures trading. The system combines advanced machine learning models, proprietary indicators, sentiment analysis, and robust risk management to create an autonomous trading application.

## ✨ Key Features

### 🎯 **Proprietary DMark Indicator**
- **Multi-component analysis**: Combines momentum, volatility, volume, microstructure, and trend signals
- **Adaptive weighting**: Adjusts based on market regime (high/low volatility, trending/ranging)
- **Confidence scoring**: Provides signal confidence levels for risk management
- **Regime detection**: Automatically identifies market conditions

### 🤖 **AI Model Zoo**
- **Grid Trading Strategies**: Linear, Fibonacci, Volatility-adjusted, Adaptive, and Kelly-based grids
- **Machine Learning Models**: Random Forest, XGBoost, LightGBM, SVM, Ridge, Lasso
- **Rule-based Strategies**: SMA Crossover, RSI, Bollinger Bands, MACD, Multi-indicator
- **Ensemble Methods**: Stacking, Adaptive weighting, Regime-based, Confidence-weighted

### 📊 **Advanced Backtesting**
- **Walk-forward validation**: Robust out-of-sample testing
- **Comprehensive metrics**: Sharpe, Sortino, Calmar, Max Drawdown, Profit Factor, VaR
- **Cost modeling**: Realistic fees, slippage, funding costs
- **Risk management**: Position sizing, stop-loss, take-profit

### 🔒 **Security & Secrets Management**
- **Encrypted credential storage**: Master password-based encryption
- **Environment variable support**: Secure API key management
- **Aster API integration**: Ready for live trading with provided credentials

### 📈 **Data Pipeline**
- **Multi-source ingestion**: Binance, OKX, X/Twitter, News APIs
- **Real-time processing**: WebSocket and REST API support
- **GCP integration**: BigQuery storage, Cloud Functions, Scheduler
- **Feature engineering**: 50+ technical and sentiment features

## 🏗️ Architecture

```
AIAster/
├── mcp_trader/                 # Core trading system
│   ├── models/                 # AI model implementations
│   │   ├── base.py            # Base model classes
│   │   ├── grid_strategies.py # Grid trading strategies
│   │   ├── ml_models.py       # Machine learning models
│   │   ├── rule_based.py      # Rule-based strategies
│   │   └── ensemble.py        # Ensemble methods
│   ├── indicators/             # Technical indicators
│   │   └── dmark.py           # Proprietary DMark indicator
│   ├── strategies/             # Trading strategies
│   │   └── dmark_strategy.py  # DMark-based strategies
│   ├── backtesting/           # Backtesting engine
│   │   └── protocol.py        # Walk-forward validation
│   ├── execution/             # Live trading
│   │   └── aster_client.py    # Aster DEX API client
│   ├── sentiment/             # Sentiment analysis
│   │   └── ingestion.py       # News/social data ingestion
│   ├── features/              # Feature engineering
│   │   └── engineering.py     # Feature pipeline
│   ├── security/              # Security utilities
│   │   └── secrets.py         # Credential management
│   └── ray_cluster.py         # Distributed computing
├── dashboard/                  # Streamlit dashboard
├── scripts/                   # Utility scripts
└── cloud_functions/           # GCP deployment
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd AIAster

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
# Run credential setup
python scripts/setup_credentials.py

# Or manually set environment variables
export ASTER_API_KEY="your-aster-api-key"
export ASTER_SECRET_KEY="your-aster-secret-key"
```

### 3. Validate Installation

```bash
# Run comprehensive validation
python scripts/validate_codebase.py

# Test DMark indicator
python scripts/test_dmark.py
```

### 4. Run Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard/app.py
```

## 🎯 DMark Indicator Usage

### Basic Usage

```python
from mcp_trader.indicators.dmark import DMarkIndicator, DMarkConfig
import pandas as pd

# Configure DMark indicator
config = DMarkConfig(
    lookback_period=20,
    momentum_period=10,
    volatility_period=14,
    strong_signal_threshold=0.7
)

# Create indicator
indicator = DMarkIndicator(config)

# Calculate signals
results = indicator.calculate(
    high=data['high'],
    low=data['low'],
    close=data['close'],
    volume=data['volume']
)

# Access results
signals = results['dmark_signal']
confidence = results['confidence']
regime = results['regime']
```

### DMark Strategy

```python
from mcp_trader.strategies.dmark_strategy import DMarkStrategy

# Create strategy
strategy = DMarkStrategy(
    min_confidence=0.6,
    max_position_size=0.25,
    stop_loss_threshold=0.02
)

# Fit and predict
strategy.fit(data)
predictions = strategy.predict(data)
signals = strategy.generate_signals(data)
```

## 🔧 Configuration

### DMark Indicator Parameters

```python
DMarkConfig(
    lookback_period=20,           # Main lookback period
    momentum_period=10,           # Momentum calculation period
    volatility_period=14,         # Volatility calculation period
    volume_period=10,             # Volume analysis period
    strong_signal_threshold=0.7,  # Strong signal threshold
    moderate_signal_threshold=0.4, # Moderate signal threshold
    weak_signal_threshold=0.2,    # Weak signal threshold
    momentum_weight=0.3,          # Momentum component weight
    volatility_weight=0.25,       # Volatility component weight
    volume_weight=0.2,            # Volume component weight
    microstructure_weight=0.15,   # Microstructure component weight
    trend_weight=0.1,             # Trend component weight
    adaptive_threshold=True       # Enable adaptive thresholds
)
```

### Strategy Parameters

```python
DMarkStrategy(
    min_confidence=0.6,           # Minimum confidence for signals
    max_position_size=0.25,       # Maximum position size (25% of capital)
    stop_loss_threshold=0.02,     # Stop loss threshold (2%)
    take_profit_threshold=0.04,   # Take profit threshold (4%)
    max_daily_trades=10,          # Maximum trades per day
    position_size_multiplier=1.0  # Position size multiplier
)
```

## 📊 Supported Assets

The system is configured for Aster DEX trading pairs:

- **BTCUSDT** - Bitcoin
- **ETHUSDT** - Ethereum  
- **ASTERUSDT** - Aster native token
- **SOLUSDT** - Solana
- **SUIUSDT** - Sui
- **PENGUUSDT** - Pengu token

## 🔒 Security Features

### Credential Management

```python
from mcp_trader.security.secrets import SecretManager

# Create secret manager with encryption
sm = SecretManager(master_password="your-password")

# Set secrets
sm.set_secret('ASTER_API_KEY', 'your-key', encrypt=True)
sm.set_secret('ASTER_SECRET_KEY', 'your-secret', encrypt=True)

# Get secrets
api_key = sm.get_secret('ASTER_API_KEY')
```

### Environment Variables

```bash
# Required for live trading
ASTER_API_KEY=your-aster-api-key
ASTER_SECRET_KEY=your-aster-secret-key

# Optional for enhanced features
CRYPTOPANIC_API_KEY=your-cryptopanic-key
NEWSAPI_API_KEY=your-newsapi-key
OPENAI_API_KEY=your-openai-key
GROQ_API_KEY=your-groq-key
```

## 🧪 Testing

### Run All Tests

```bash
# Comprehensive validation
python scripts/validate_codebase.py

# DMark indicator tests
python scripts/test_dmark.py

# Backtesting
python scripts/run_backtest.py
```

### Test DMark Indicator

```python
# Generate sample data
data = generate_sample_data(days=30)

# Test indicator
indicator = DMarkIndicator()
results = indicator.calculate(data['high'], data['low'], data['close'], data['volume'])

# Analyze results
print(f"Signal range: {results['dmark_signal'].min():.3f} to {results['dmark_signal'].max():.3f}")
print(f"Confidence range: {results['confidence'].min():.3f} to {results['confidence'].max():.3f}")
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Return Metrics**: Total return, annualized return, Sharpe ratio, Sortino ratio
- **Risk Metrics**: Max drawdown, VaR, CVaR, tail ratio
- **Trade Metrics**: Win rate, profit factor, average win/loss
- **Risk Management**: Position sizing, exposure, turnover

## 🚀 Deployment

### Local Development

```bash
# Start dashboard
streamlit run dashboard/app.py

# Run backtesting
python scripts/run_backtest.py

# Test live trading (with paper trading)
python scripts/live_trading.py
```

### GCP Deployment

```bash
# Deploy to GCP
./scripts/deploy_gcp.sh

# Setup BigQuery
python scripts/setup_gcp.py

# Deploy Cloud Functions
python scripts/deploy_functions.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python scripts/validate_codebase.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Support

For questions and support:

1. Check the documentation
2. Run validation: `python scripts/validate_codebase.py`
3. Test components: `python scripts/test_dmark.py`
4. Open an issue on GitHub

---

**Built with ❤️ for the Aster ecosystem**
