# Aster Data Structure Guide

## 🎯 Overview

A dedicated, optimized data structure exclusively for **Aster DEX perpetual contracts**, designed for robust backtesting and live trading agent deployment.

## 📊 Current Status

### Data Assets
- **Total Assets**: 97 crypto perpetuals
- **High Quality (>80%)**: 82 assets
- **Perfect Quality (100%)**: 37 assets
- **Total Historical Candles**: 194,097
- **Average Data Quality**: 91.59%

### Top 20 Trading Universe
```
ADA, ALGO, ANKR, ATOM, AVAX, BTC, CELO, CHZ, CRO, CRV, 
DOGE, DOT, ETH, FTM, GALA, GRT, HBAR, IOTX, LEO, LINK
```

## 🏗️ Architecture

### 1. Asset Registry
**Location**: `data/aster_asset_registry.json`

Centralized metadata for all Aster DEX assets:
- Symbol and asset type (crypto_perp/stock_perp)
- Trading parameters (leverage, tick size, fees)
- Data quality metrics
- Date ranges and candle counts

**Access**:
```python
from mcp_trader.data.aster_asset_manager import AsterAssetRegistry

registry = AsterAssetRegistry()
registry.load_registry()

# Get asset info
asset = registry.get_asset('BTC')
print(f"Leverage: {asset.max_leverage}x")
print(f"Quality: {asset.data_quality_score:.2%}")
print(f"Candles: {asset.total_candles}")
```

### 2. Data Structure
**Location**: `mcp_trader/data/aster_asset_manager.py`

Core data management system:
- **AsterAssetMetadata**: Asset information
- **AsterAssetRegistry**: Registry management
- **AsterDataStructure**: Data loading and validation

**Features**:
- Fast data loading with LRU caching
- Standardized OHLCV format
- Data quality scoring
- Multi-format export (Parquet, HDF5)

### 3. Backtesting Loader
**Location**: `mcp_trader/backtesting/aster_backtest_loader.py`

Optimized for strategy backtesting:
- Fast historical data loading
- Trading universe selection
- Train/test splitting
- Timestamp alignment
- Timeframe resampling

## 🚀 Quick Start

### Step 1: Validate Data
```bash
python scripts/validate_aster_data.py
```

### Step 2: Load Data for Backtesting
```python
from mcp_trader.backtesting.aster_backtest_loader import create_backtest_loader

# Create loader
loader = create_backtest_loader(min_quality=0.85)

# Get trading universe
symbols = loader.get_trading_universe(top_n=20)
print(f"Trading {len(symbols)} assets: {symbols}")

# Load historical data
data = loader.load_backtest_data(
    symbols=symbols,
    start_date='2025-08-01',
    end_date='2025-10-15'
)

# Access data for each asset
for symbol, df in data.items():
    print(f"{symbol}: {len(df)} candles")
    print(df.head())
```

### Step 3: Train/Test Split
```python
# Split data for strategy optimization
train_data, test_data = loader.get_train_test_split(
    symbols=symbols,
    train_ratio=0.7
)

# Optimize strategy on train set
# Validate on test set
```

### Step 4: Aligned Multi-Asset Data
```python
# For portfolio/multi-asset strategies
aligned_data = loader.get_aligned_data(
    symbols=['BTC', 'ETH', 'SOL'],
    start_date='2025-09-01'
)

# All assets now have matching timestamps
```

## 📁 Directory Structure

```
data/
├── aster_asset_registry.json          # Asset metadata registry
├── aster_backtesting_dataset.h5       # HDF5 format (optional)
└── historical/
    └── aster_perps/
        ├── crypto/                     # Crypto perpetuals data
        │   ├── BTC_consolidated.parquet
        │   ├── ETH_consolidated.parquet
        │   └── ... (97 assets)
        └── stocks/                     # Stock perpetuals (future)
            ├── AAPL_stock_perp.parquet
            └── ...
```

## 🔑 Key Features

### 1. Data Quality Scoring
Each asset has a quality score (0-1) based on:
- **Completeness**: No missing values
- **Consistency**: Valid OHLC relationships
- **Integrity**: No zero/negative prices
- **Continuity**: Minimal gaps
- **Size**: Sufficient historical depth

### 2. Rate Limiting & Resilience
- Automatic retry logic
- Configurable delays per exchange
- Graceful error handling
- Multi-source fallback (Binance, Kraken, KuCoin, CoinGecko)

### 3. Optimized for Trading
- **Leverage**: Up to 50x on Aster DEX
- **Fees**: 0.02% maker, 0.05% taker
- **Funding**: 8-hour intervals
- **24/7 Trading**: Including stock perpetuals

### 4. Backtesting Ready
- Standardized timestamp indexing
- Consistent OHLCV format
- Train/test splitting
- Multi-asset alignment
- Timeframe resampling

## 🎯 Recommended Trading Universe

### Tier 1: Perfect Quality (100%)
Best assets for backtesting and live trading:
```
BTC, ETH, ADA, ALGO, ANKR, ATOM, AVAX, CELO, CHZ, CRO,
CRV, DOGE, DOT, FTM, GALA, GRT, HBAR, IOTX, LEO, LINK,
LTC, MANA, MATIC, NEAR, ONE, ROSE, SAND, SHIB, SOL, TRX,
UNI, VET, XLM, XRP, ZEC, ZIL, ZRX
```
*37 assets with 100% data quality*

### Tier 2: High Quality (>90%)
Excellent for diversification:
```
AAVE, APT, AXS, BAND, BAT, BCH, BNB, CAKE, COMP, EOS,
ETC, FIL, FLOW, ICX, KDA, KSM, LRC, LUNA, MKR, NEO,
OCEAN, OMG, ONT, QNT, RVN, SNX, STORJ, THETA, TWT,
WAVES, XEC, XTZ, YFI
```
*45 additional assets with >90% quality*

## 📈 Data Specifications

### Timeframe
- **Base**: 1-hour candles
- **Resample**: 4h, 1d, 1w support

### Coverage
- **Start Date**: 2025-07-23 (varies by asset)
- **End Date**: 2025-10-15 (current)
- **Typical Range**: ~2,000 hourly candles per asset

### Format
```python
# DataFrame structure
timestamp (index)  | open | high | low | close | volume
2025-10-15 06:00  | ...  | ...  | ... | ...   | ...
```

## 🛠️ Maintenance Scripts

### Build/Rebuild Registry
```bash
python scripts/build_aster_registry_simple.py
```

### Validate Data
```bash
python scripts/validate_aster_data.py
```

### Collect New Data
```bash
python scripts/collect_aster_perps_data.py
```

## 🔄 Updating Data

To add new assets or update existing data:

1. **Add new crypto perpetuals**:
   - Place `{SYMBOL}_consolidated.parquet` in `data/historical/aster_perps/crypto/`
   - Run `python scripts/build_aster_registry_simple.py`

2. **Add stock perpetuals**:
   - Place `{SYMBOL}_stock_perp.parquet` in `data/historical/aster_perps/stocks/`
   - Rebuild registry

3. **Update existing data**:
   - Replace parquet files
   - Rebuild registry to update metadata

## ⚡ Performance Tips

1. **Use caching**: Loader caches up to 50 assets in memory
2. **Limit universe**: Start with top 20-30 assets
3. **Date filtering**: Filter early to reduce memory usage
4. **HDF5 export**: Use for very large backtests

## 🚨 Important Notes

### Rate Limits
- **Binance**: Requires VPN (Iceland, Singapore, etc.)
- **Kraken**: 0.3s delay between requests
- **KuCoin**: 0.2s delay between requests
- **CoinGecko**: 2s delay, 30 assets/minute

### Data Quality
- Assets with <70% quality: Use with caution
- Missing data: Check logs for unavailable assets
- Resampling: May reduce data points

### Trading Considerations
- **Leverage**: High leverage = high risk
- **Funding fees**: Paid every 8 hours
- **Slippage**: Consider in backtests
- **Market hours**: 24/7 for crypto, extended for stocks

## 📚 Next Steps

1. **Backtest strategies** using the data loader
2. **Optimize parameters** with train/test split
3. **Deploy trading agents** to Aster DEX
4. **Monitor performance** in live trading
5. **Iterate and improve** based on results

## 🤝 Integration with Trading Agents

The data structure is designed to integrate seamlessly with:
- `mcp_trader/strategies/` - Strategy implementations
- `mcp_trader/execution/` - Order execution
- `mcp_trader/risk/` - Risk management
- `mcp_trader/backtesting/` - Backtesting engines

## ✅ Checklist

- [x] Data structure created
- [x] Asset registry built (97 assets)
- [x] Data quality validated (91.59% avg)
- [x] Backtesting loader implemented
- [x] Documentation completed
- [ ] Backtest first strategy
- [ ] Deploy first trading agent
- [ ] Monitor live performance

---

**Status**: ✅ **READY FOR BACKTESTING AND TRADING**

**Last Updated**: October 15, 2025

**Total Assets**: 97 crypto perpetuals  
**Data Quality**: 91.59% average  
**Perfect Quality Assets**: 37  
**Recommended Universe**: 20 top assets

