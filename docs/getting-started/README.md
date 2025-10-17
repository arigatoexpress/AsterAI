# Getting Started with Rari Trade

Welcome to Rari Trade! This guide will walk you through setting up and running your AI trading platform step by step.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: Version 3.11 or higher
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 10GB free space for data and models
- **Internet**: Stable broadband connection

### Hardware Recommendations
- **CPU**: Intel i5/AMD Ryzen 5 or better
- **GPU**: NVIDIA RTX 3060 or better (optional, for accelerated training)
- **Storage**: SSD recommended for faster data processing

## Step 1: Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/rari-trade.git
cd rari-trade
```

### Create Virtual Environment
```bash
# Windows
python -m venv rari_trade_env
rari_trade_env\Scripts\activate

# macOS/Linux
python -m venv rari_trade_env
source rari_trade_env/bin/activate
```

### Install Dependencies
```bash
# Install core requirements
pip install -r requirements.txt

# Optional: Install GPU acceleration
pip install -r requirements-gpu.txt
```

## Step 2: Configuration

### API Keys Setup
1. Create a copy of the template configuration:
```bash
cp config/api_keys_template.json config/api_keys.json
```

2. Edit the configuration file with your API credentials:
```json
{
  "exchange_credentials": {
    "api_key": "your_api_key_here",
    "api_secret": "your_api_secret_here",
    "testnet": true
  }
}
```

### Trading Parameters
Configure your risk management settings:
```bash
cp config/trading_config_template.json config/trading_config.json
```

## Step 3: Data Collection

### Download Historical Data
```bash
python scripts/collect_historical_data.py
```

This will:
- Connect to exchange APIs
- Download price and volume data
- Calculate technical indicators
- Store data in organized format

**Expected Output**: Data files saved to `data/historical/` directory

## Step 4: Model Training

### Train AI Models
```bash
python scripts/train_cpu_fallback.py
```

**What happens during training:**
1. Loads historical market data
2. Calculates 41 technical indicators
3. Trains ensemble of 7 AI models
4. Validates performance metrics
5. Saves trained models

**Expected Duration**: 10-30 minutes depending on hardware

## Step 5: Paper Trading Validation

### Run Paper Trading Test
```bash
python scripts/setup_paper_trading.py
```

**Validation Criteria:**
- Sharpe Ratio > 1.5 (risk-adjusted returns)
- Win Rate > 55% (successful trades)
- Maximum Drawdown < 10% (capital preservation)
- Minimum 50 trades (statistical significance)

**Expected Duration**: 7 days of simulated trading

## Step 6: Live Trading

### Start Live Trading (After Validation)
```bash
python scripts/deploy_live_trading.py
```

**Safety Features Activated:**
- Emergency stop loss at 10% drawdown
- Daily loss limit of 3%
- Position size limits
- Real-time risk monitoring

## Step 7: Monitoring

### Access Dashboard
The system includes a web dashboard for monitoring:

```bash
# Local dashboard
python -m streamlit run dashboard/unified_trading_dashboard.py

# Visit: http://localhost:8501
```

### Dashboard Features
- Real-time portfolio valuation
- Active position tracking
- Performance metrics
- Risk monitoring
- Trade history

## Understanding the Output

### Key Metrics to Monitor
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Gross profit divided by gross loss
- **Daily P&L**: Profit and loss tracking

### Common Indicators
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100)
- **MACD**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility bands around price
- **Volume**: Trading volume analysis

## Troubleshooting

### Common Issues
1. **API Connection Failed**: Check internet connection and API keys
2. **Out of Memory**: Reduce batch size in configuration
3. **GPU Not Detected**: Install CUDA drivers or use CPU mode
4. **Data Download Slow**: Check exchange rate limits

### Getting Help
- Check the [troubleshooting guide](../troubleshooting/)
- Review [logs/](logs/) for error messages
- Open an issue on GitHub

## Next Steps

Once running successfully:
1. Monitor performance daily
2. Adjust risk parameters based on results
3. Consider scaling capital (see capital scaling guide)
4. Explore advanced features (sentiment analysis, multi-asset)

## Safety First

Remember:
- Start with small amounts you're comfortable losing
- Never invest more than you can afford
- Monitor the system regularly
- Have emergency stop procedures ready

---

*Ready to start your algorithmic trading journey? The system is designed to grow from $100 to $10K+ through intelligent automation.*
