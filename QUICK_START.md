# 🚀 AsterAI Quick Start Guide

Get your AI trading system up and running in **30 minutes**!

---

## Step 1: Install Dependencies (5 minutes)

```powershell
# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA for RTX 5070 Ti
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 2: Setup API Keys (5 minutes)

```powershell
# Interactive setup - follow the prompts
python scripts/setup_api_keys.py --interactive
```

**Minimum Required:**
- ASTER_API_KEY (for trading)
- ASTER_SECRET_KEY (for trading)

**Recommended (all free):**
- Alpha Vantage API key → https://www.alphavantage.co/support/#api-key
- FRED API key → https://fred.stlouisfed.org/docs/api/api_key.html

---

## Step 3: Verify GPU Setup (2 minutes)

```powershell
python scripts/verify_gpu_setup.py
```

**Look for:**
- ✓ CUDA Available: True
- ✓ GPU: NVIDIA GeForce RTX 5070 Ti
- ✓ Total Memory: 16.00 GB

---

## Step 4: Test Data Pipeline (3 minutes)

```powershell
python scripts/test_data_pipeline.py
```

**This tests:**
- CoinGecko API (crypto data)
- Yahoo Finance (stocks)
- Your configured API keys

---

## Step 5: Collect Historical Data (1-2 hours, runs in background)

```powershell
# Collect data from 2024 onwards for top assets
python scripts/collect_historical_data.py --start-date 2024-01-01 --priority 2
```

**What this does:**
- Fetches 2024+ data for 40+ assets
- Stores in efficient Parquet format
- Validates data quality
- Creates integrity checksums

**Let this run in the background while you continue...**

---

## Step 6: Start Dashboard (2 minutes)

```powershell
python dashboard/aster_trader_dashboard.py --port 8001
```

Open browser: http://localhost:8001

**You'll see:**
- Real-time market data
- Multi-asset overview
- AI learning status
- Position management

---

## Step 7: Analyze Data (5 minutes)

Create `scripts/quick_analysis.py`:

```python
import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.historical_collector import HistoricalDataCollector
from mcp_trader.analysis.data_analyzer import DataAnalyzer

async def main():
    collector = HistoricalDataCollector()
    analyzer = DataAnalyzer()
    
    # Get summary of collected data
    summary = collector.get_collection_summary()
    print(f"\n✓ Collected {summary['total_datasets']} datasets")
    print(f"✓ Total records: {summary['total_records']:,}")
    
    # Analyze BTC
    btc = collector.asset_universe.get_asset('BTC')
    if btc:
        df = collector.load_dataset(btc)
        if df is not None:
            report = analyzer.generate_comprehensive_report(df, 'BTC')
            print(f"\n📊 BTC Analysis:")
            print(f"  Mean Price: ${report['statistics']['mean']:,.2f}")
            print(f"  Volatility: {report['statistics']['std_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {report['statistics']['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {report.get('max_drawdown', 0):.2%}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```powershell
python scripts/quick_analysis.py
```

---

## Step 8: Train Your First Model (2-4 hours on RTX 5070 Ti)

See `DATA_PIPELINE_GUIDE.md` Phase 6 for detailed ML training instructions.

**Quick start:**
```powershell
# Train LSTM price predictor
python scripts/train_lstm.py

# Train RL trading agent
python scripts/train_ppo_agent.py
```

---

## What You've Built

✅ **Data Pipeline**
- Multi-source data collection (7+ APIs)
- Parquet storage (10x faster than CSV)
- Data validation & integrity checks
- Automated redundancy & fallbacks

✅ **Analysis Suite**
- Statistical analysis
- Correlation detection
- Market regime identification
- Support/resistance levels
- Momentum indicators

✅ **ML Infrastructure**
- GPU-optimized PyTorch setup
- Mixed precision training
- Model checkpointing
- Performance monitoring

✅ **Dashboard**
- Real-time market overview
- AI strategy adaptation
- Position management
- Analytics & backtesting

---

## Daily Workflow

### Morning (5 minutes)
```powershell
# Update data
python scripts/update_historical_data.py

# Check integrity
python scripts/collect_historical_data.py --verify
```

### Trading Day
```powershell
# Start dashboard
python dashboard/aster_trader_dashboard.py --port 8001

# Monitor at http://localhost:8001
```

### Evening (as needed)
```powershell
# Retrain models with new data
python scripts/train_lstm.py

# Analyze performance
python scripts/analyze_trading_performance.py
```

---

## Troubleshooting

### "CUDA not available"
```powershell
# Reinstall PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "API rate limit exceeded"
```powershell
# The pipeline automatically retries with exponential backoff
# Just wait a few minutes and it will continue
```

### "Out of memory" (GPU)
```python
# In your training script, reduce batch size:
BATCH_SIZE = 256  # Instead of 512
```

### "Port 8000 already in use"
```powershell
# Use different port
python dashboard/aster_trader_dashboard.py --port 8001
```

---

## Next Steps

1. **Read Full Guide**: `DATA_PIPELINE_GUIDE.md`
2. **GPU Optimization**: `RTX_5070Ti_ML_SETUP.md`
3. **Security Setup**: `SECURITY.md`
4. **Deploy to Cloud**: `README_GCP.md`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `scripts/setup_api_keys.py` | Configure API keys |
| `scripts/verify_gpu_setup.py` | Verify GPU/CUDA |
| `scripts/test_data_pipeline.py` | Test data sources |
| `scripts/collect_historical_data.py` | Collect historical data |
| `dashboard/aster_trader_dashboard.py` | Start web dashboard |
| `mcp_trader/data/multi_source_pipeline.py` | Data pipeline core |
| `mcp_trader/analysis/data_analyzer.py` | Analysis tools |
| `mcp_trader/models/` | ML models |

---

## Support

- **Documentation**: See `docs/` directory
- **Examples**: See `scripts/` directory
- **Issues**: Check error logs in `logs/`

---

**You're ready to trade with AI! 🎯📈🚀**



