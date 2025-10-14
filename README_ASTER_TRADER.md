# 🚀 Rari Trade AI - Aster Autonomous Trader

**AI-powered autonomous trading system focused on maximizing profits through intelligent grid trading and risk management on Aster DEX.**

## 🎯 Overview

This system is designed from first principles to autonomously trade volatile assets on Aster DEX using advanced algorithms:

- **Grid Trading**: Automated buy/sell levels for ranging markets
- **Volatility Strategies**: Capitalize on price swings in high-volatility assets
- **Risk Management**: Multi-layered protection against losses
- **Real-time Execution**: Direct integration with Aster DEX API

## 🏗️ Architecture

### Core Components

```
mcp_trader/
├── trading/
│   ├── autonomous_trader.py    # Main trading engine
│   ├── strategies/
│   │   ├── grid_strategy.py    # Grid trading logic
│   │   └── volatility_strategy.py  # Volatility-based trading
│   └── grid/
├── execution/
│   └── aster_client.py         # Aster DEX API client (complete)
├── risk/volatility/
│   └── risk_manager.py         # Risk management system
├── data/
│   └── aster_feed.py           # Real-time data feed
├── config.py                   # Aster-focused configuration
└── __init__.py
```

### Trading Modes

1. **Grid Trading** (`grid`): Creates automated buy/sell grids for ranging markets
2. **Volatility Trading** (`volatility`): Exploits price swings in volatile assets
3. **Hybrid** (`hybrid`): Combines both strategies for optimal performance

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set API credentials
export ASTER_API_KEY="your_aster_api_key"
export ASTER_API_SECRET="your_aster_api_secret"
```

### 2. Run the Trader

```bash
# Start autonomous trading in hybrid mode
python run_aster_trader.py hybrid

# Grid trading only
python run_aster_trader.py grid

# Test mode (no real trades)
python run_aster_trader.py hybrid --test
```

### 3. Monitor via Dashboard

```bash
# Run the monitoring dashboard
python dashboard/aster_trader.py
```

## 🎯 Trading Assets

The system focuses on high-volatility assets optimal for automated strategies:

| Asset | Volatility | Strategy | Leverage |
|-------|------------|----------|----------|
| BTCUSDT | Medium | Conservative | 50x |
| ETHUSDT | Medium | Conservative | 50x |
| SOLUSDT | High | Grid + Volatility | 25x |
| SUIUSDT | High | Grid + Volatility | 25x |
| ASTERUSDT | Medium | Grid | 10x |
| PENGUUSDT | Extreme | Volatility | 5x |

## 🛡️ Risk Management

### Multi-Layered Protection

1. **Portfolio Level**: Max 10% portfolio risk per trade
2. **Position Level**: Max 5% per individual position
3. **Daily Loss Limits**: 15% maximum daily drawdown
4. **Volatility Adjustment**: Position sizes scale with market volatility
5. **Emergency Stops**: Automatic shutdown on critical errors

### Grid Trading Features

- **Adaptive Spacing**: Grid levels adjust based on volatility
- **Profit Taking**: Automated profit realization at target levels
- **Rebalancing**: Dynamic grid repositioning during trends
- **Liquidity Checks**: Ensures sufficient order book depth

### Volatility Trading Features

- **Momentum Detection**: Identifies trending moves
- **Mean Reversion**: Capitalizes on price pullbacks
- **Dynamic Sizing**: Position sizes based on volatility levels
- **Time-based Exits**: Prevents holding positions too long

## 📊 Performance Monitoring

### Real-time Metrics

- Portfolio value and P&L tracking
- Win rate and profit factor
- Sharpe ratio and volatility metrics
- Drawdown monitoring
- Trade execution statistics

### Dashboard Features

- Live position monitoring
- Performance charts
- Risk metrics display
- Trading activity log
- Emergency controls

## 🔧 Configuration

### Environment Variables

```bash
# Required
ASTER_API_KEY=your_api_key
ASTER_API_SECRET=your_api_secret

# Optional
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Trading Parameters

Edit `mcp_trader/config.py` to adjust:

```python
# Risk limits
max_portfolio_risk = 0.1      # 10%
max_single_position_risk = 0.05  # 5%
max_daily_loss = 0.15         # 15%

# Grid settings
grid_levels = 10              # Number of grid levels
grid_spacing_percent = 2.0    # 2% spacing
grid_position_size_usd = 50.0 # $50 per level

# Volatility settings
stop_loss_threshold = 0.03    # 3%
take_profit_threshold = 0.05  # 5%
```

## 🔍 Technical Details

### Grid Trading Algorithm

1. **Initialization**: Creates buy/sell levels around current price
2. **Level Triggers**: Executes trades when price hits grid levels
3. **Profit Realization**: Takes profits at opposite levels
4. **Rebalancing**: Adjusts grid during significant price moves

### Risk Assessment

Each trade undergoes multi-stage risk evaluation:

1. **Pre-trade Analysis**: Portfolio impact, correlation, liquidity
2. **Volatility Adjustment**: Position sizing based on market conditions
3. **Execution Checks**: Slippage and order book validation
4. **Post-trade Monitoring**: Real-time P&L and drawdown tracking

### Data Pipeline

- **Real-time Feeds**: WebSocket connections for live prices
- **Order Book Data**: Depth analysis for liquidity assessment
- **Trade History**: Volume and price action analysis
- **Volatility Calculation**: Rolling volatility metrics

## 🚨 Safety Features

### Emergency Controls

- **Manual Stop**: Immediate halt via dashboard or API
- **Automatic Shutdown**: Triggers on excessive losses or errors
- **Circuit Breakers**: Temporary halts during extreme volatility
- **Position Limits**: Hard caps on position sizes

### Error Handling

- **API Resilience**: Automatic retry on connection failures
- **Data Validation**: Comprehensive input validation
- **Logging**: Detailed audit trail of all actions
- **Recovery**: Graceful restart after failures

## 📈 Performance Optimization

### Profit Maximization Strategies

1. **Asset Selection**: Focus on high-volatility assets with grid potential
2. **Timing Optimization**: Market regime detection for strategy switching
3. **Size Optimization**: Kelly Criterion for position sizing
4. **Cost Management**: Minimize trading fees through smart execution

### Backtesting Integration

```bash
# Run backtests (when implemented)
python scripts/backtest_grid_strategy.py
python scripts/backtest_volatility_strategy.py
```

## 🔮 Future Enhancements

### Planned Features

- **Machine Learning Models**: LSTM and Transformer-based prediction
- **Cross-Asset Arbitrage**: Spot-perpetual arbitrage opportunities
- **Sentiment Integration**: Social media sentiment analysis
- **Advanced Grid Types**: Arithmetic, geometric, and custom grids

### Scaling Considerations

- **Multi-Asset Expansion**: Support for additional Aster trading pairs
- **Portfolio Optimization**: Modern portfolio theory integration
- **High-Frequency Trading**: Sub-second execution for optimal fills
- **Cloud Deployment**: GCP/AWS deployment with auto-scaling

## ⚠️ Important Disclaimers

### Risk Warnings

- **High Risk**: Trading cryptocurrencies involves substantial risk of loss
- **Volatility**: Assets like PENGU can experience extreme price swings
- **No Guarantees**: Past performance does not predict future results
- **Test First**: Always test strategies in simulation before live trading

### Regulatory Compliance

- **Your Responsibility**: Ensure compliance with local regulations
- **Tax Implications**: Track and report trading activity for tax purposes
- **AML/KYC**: Comply with Aster DEX requirements

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository>
cd rari-trade-ai
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run linter
python -m flake8 mcp_trader/
```

### Code Standards

- **Type Hints**: All functions use proper type annotations
- **Documentation**: Comprehensive docstrings for all modules
- **Testing**: Unit tests for critical components
- **Logging**: Structured logging throughout the system

## 📞 Support

### Getting Help

- **Documentation**: Check this README and inline code documentation
- **Logs**: Review `aster_trader.log` for detailed system activity
- **Dashboard**: Use the web dashboard for real-time monitoring
- **Testing**: Run in `--test` mode to validate setup

### Common Issues

1. **API Connection Failed**: Verify ASTER_API_KEY and ASTER_API_SECRET
2. **No Positions Opening**: Check risk limits and market conditions
3. **High Slippage**: Reduce position sizes or use limit orders
4. **Dashboard Not Loading**: Ensure port 8501 is available

---

## 🎯 Mission Statement

**To create the most profitable autonomous trading system on Aster DEX by combining advanced algorithms, rigorous risk management, and continuous optimization.**

*Built for serious traders who demand institutional-grade automation with retail accessibility.* 🚀
