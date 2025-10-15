# AsterAI Data Pipeline & ML Training Guide

## 🎯 Your Complete Setup Plan

This guide walks you through setting up your complete AI trading system with historical data, analysis, and ML training on your RTX 5070 Ti.

---

## Phase 1: API Keys & Data Sources Setup (15 minutes)

### Step 1: Get Free API Keys

1. **Alpha Vantage** (Recommended - Stocks & Technical Indicators)
   - Visit: https://www.alphavantage.co/support/#api-key
   - Free: 5 calls/min, 500/day
   - Use for: Stock data, technical indicators

2. **FRED** (Recommended - Economic Data)
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Free: Unlimited
   - Use for: GDP, inflation, interest rates

3. **Finnhub** (Optional - News & Sentiment)
   - Visit: https://finnhub.io/register
   - Free: 60 calls/min
   - Use for: Stock news, sentiment

4. **NewsAPI** (Optional - Market News)
   - Visit: https://newsapi.org/register
   - Free: 1000 requests/day
   - Use for: Market-moving news

### Step 2: Configure API Keys

```powershell
# Interactive setup (recommended)
python scripts/setup_api_keys.py --interactive

# Or set environment variables
$env:ASTER_API_KEY="your_aster_key"
$env:ASTER_SECRET_KEY="your_aster_secret"
$env:ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
$env:FRED_API_KEY="your_fred_key"
```

### Step 3: Test Data Sources

```powershell
# Test all configured APIs
python scripts/test_data_pipeline.py
```

**Expected Output:**
- ✓ CoinGecko: Working (no key needed)
- ✓ Yahoo Finance: Working (no key needed)
- ✓ Alpha Vantage: Working (if key configured)
- ✓ FRED: Working (if key configured)

---

## Phase 2: GPU Setup & Verification (30 minutes)

### Step 1: Install NVIDIA Drivers & CUDA

```powershell
# Check if NVIDIA driver is installed
nvidia-smi

# If not, download latest driver from:
# https://www.nvidia.com/Download/index.aspx

# Install CUDA Toolkit 12.1+
# https://developer.nvidia.com/cuda-downloads
```

### Step 2: Install PyTorch with CUDA

```powershell
# Install PyTorch 2.x with CUDA 12.1 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify GPU Setup

```powershell
python scripts/verify_gpu_setup.py
```

**Expected Output:**
```
✓ PyTorch: 2.x.x
✓ CUDA Available: True
✓ CUDA Version: 12.1
✓ GPU: NVIDIA GeForce RTX 5070 Ti
✓ Total Memory: 16.00 GB
✓ 4th Gen Tensor Cores Available
✓ TF32 Acceleration Supported
```

See `RTX_5070Ti_ML_SETUP.md` for detailed GPU optimization guide.

---

## Phase 3: Historical Data Collection (1-2 hours)

### Step 1: Collect Data from 2024 Onwards

```powershell
# Collect all priority 1 & 2 assets from 2024-01-01
python scripts/collect_historical_data.py --start-date 2024-01-01 --priority 2

# This will collect:
# - Top 15 cryptocurrencies (BTC, ETH, SOL, etc.)
# - Mag 7 stocks (AAPL, MSFT, NVDA, etc.)
# - Major indices (SPY, QQQ)
# - Commodities (Gold, Silver, Uranium, Lithium)
# - Economic indicators (GDP, CPI, Fed Funds, etc.)
```

**What Happens:**
1. Fetches historical OHLCV data from multiple sources
2. Validates data quality and integrity
3. Stores in Parquet format (5-10x faster than CSV)
4. Creates metadata with checksums for verification
5. Rate-limited to avoid API blocks

**Storage:**
- Data stored in: `data/historical/`
- Structure:
  ```
  data/historical/
  ├── crypto/
  │   ├── btc.parquet
  │   ├── eth.parquet
  │   └── ...
  ├── stocks/
  │   ├── aapl.parquet
  │   ├── msft.parquet
  │   └── ...
  ├── commodities/
  │   └── gld.parquet
  ├── economic/
  │   └── gdp.parquet
  └── metadata/
      ├── btc_meta.json
      └── ...
  ```

### Step 2: Verify Data Integrity

```powershell
# Verify all collected datasets
python scripts/collect_historical_data.py --verify
```

### Step 3: View Collection Summary

```powershell
# See what you've collected
python scripts/collect_historical_data.py --summary
```

---

## Phase 4: Data Analysis & Exploration (1 hour)

### Step 1: Statistical Analysis

Create `scripts/analyze_data.py`:

```python
import asyncio
import pandas as pd
from pathlib import Path
from mcp_trader.data.historical_collector import HistoricalDataCollector
from mcp_trader.analysis.data_analyzer import DataAnalyzer

async def main():
    collector = HistoricalDataCollector()
    analyzer = DataAnalyzer()
    
    # Load BTC data
    btc_asset = collector.asset_universe.get_asset('BTC')
    btc_df = collector.load_dataset(btc_asset)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(btc_df, 'BTC')
    
    print("="*70)
    print("BTC Analysis Report")
    print("="*70)
    print(f"\nStatistics:")
    for key, value in report['statistics'].items():
        print(f"  {key:20s}: {value}")
    
    print(f"\nSupport Levels: {report['support_resistance']['support']}")
    print(f"Resistance Levels: {report['support_resistance']['resistance']}")
    print(f"Max Drawdown: {report['max_drawdown']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```powershell
python scripts/analyze_data.py
```

### Step 2: Correlation Analysis

```python
# Find correlated assets
data_dict = {}
for symbol in ['BTC', 'ETH', 'AAPL', 'NVDA', 'SPY']:
    asset = collector.asset_universe.get_asset(symbol)
    df = collector.load_dataset(asset)
    if df is not None:
        data_dict[symbol] = df

# Calculate correlations
correlations = analyzer.find_correlations(data_dict, threshold=0.7)

print("\nHighly Correlated Assets:")
for asset1, asset2, corr in correlations:
    print(f"  {asset1} <-> {asset2}: {corr:.3f}")
```

### Step 3: Market Regime Detection

```python
# Detect bull/bear markets
btc_with_regime = analyzer.detect_regime_changes(btc_df)

# See regime distribution
print("\nMarket Regimes:")
print(btc_with_regime['regime'].value_counts())
```

---

## Phase 5: Data Visualization (Setup)

Your dashboard templates are already created in `dashboard/templates/`:
- `market_overview.html` - Main market dashboard
- `ai_learning.html` - AI strategy adaptation
- `positions.html` - Position management
- `analytics.html` - Backtesting & analytics

To integrate your collected data into the dashboard, you'll run:

```powershell
python dashboard/aster_trader_dashboard.py --port 8001
```

Then visit: http://localhost:8001

---

## Phase 6: ML Model Training (Your RTX 5070 Ti)

### Model 1: LSTM Price Predictor

Create `scripts/train_lstm.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from mcp_trader.data.historical_collector import HistoricalDataCollector

# Enable RTX 5070 Ti optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=256, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load data
collector = HistoricalDataCollector()
btc = collector.asset_universe.get_asset('BTC')
df = collector.load_dataset(btc)

# Prepare data
# ... (feature engineering, normalization)

# Create model
device = torch.device('cuda')
model = LSTMPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training loop
for epoch in range(100):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    print(f"Epoch {epoch+1}/100, Loss: {loss.item():.6f}")

# Save model
torch.save(model.state_dict(), 'models/lstm_btc.pth')
```

**Expected Training Time:** ~2 hours on RTX 5070 Ti

### Model 2: PPO Reinforcement Learning Agent

Create `scripts/train_ppo_agent.py`:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from mcp_trader.models.reinforcement_learning.trading_agents import TradingEnv

# Create trading environment
env = DummyVecEnv([lambda: TradingEnv(df=btc_df)])

# Create PPO agent (optimized for RTX 5070 Ti)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,  # Large batch size for GPU
    batch_size=512,
    n_epochs=10,
    device='cuda',
    verbose=1
)

# Train
model.learn(total_timesteps=1_000_000)

# Save
model.save("models/ppo_trading_agent")
```

**Expected Training Time:** ~4 hours on RTX 5070 Ti

### Model 3: Ensemble Strategy

```python
# Combine LSTM predictions with RL actions
ensemble = EnsembleModel([lstm_model, ppo_model])
```

---

## Phase 7: Live Trading Integration

Once models are trained:

```python
# In your live trading system
from mcp_trader.ai.adaptive_trading_agent import AdaptiveTradingAgent
from mcp_trader.execution.aster_client import AsterClient

# Initialize
agent = AdaptiveTradingAgent(config=trading_config)
client = AsterClient(api_key, secret_key)

# Run
await agent.run_trading_loop()
```

---

## Data Update Strategy

### Daily Updates

```powershell
# Update all datasets with latest data
python scripts/update_historical_data.py
```

This will:
1. Check last update time for each asset
2. Fetch only new data since last update
3. Append to existing Parquet files
4. Update metadata and checksums

### Automated Schedule

```powershell
# Windows Task Scheduler
schtasks /create /tn "AsterAI Daily Update" /tr "python scripts/update_historical_data.py" /sc daily /st 00:00
```

---

## Data Integrity & Backup

### Verification

```powershell
# Daily integrity check
python scripts/collect_historical_data.py --verify
```

### Backup Strategy

```powershell
# Backup to cloud storage
python scripts/backup_data.py --destination gcs://asterai-data-backup/

# Or local backup
python scripts/backup_data.py --destination D:/Backups/AsterAI/
```

---

## Performance Benchmarks

### Expected Performance on RTX 5070 Ti

| Model | Batch Size | Training Time | Inference |
|-------|------------|---------------|-----------|
| LSTM (3 layers) | 512 | ~2 hours | <1ms |
| PPO RL | 2048 | ~4 hours | ~2ms |
| Transformer | 128 | ~3 hours | ~3ms |

### Data Pipeline Performance

| Operation | Speed |
|-----------|-------|
| Parquet Read | 10x faster than CSV |
| Data Validation | ~1000 records/sec |
| API Fetch | Rate-limited by source |
| GPU Data Transfer | ~15 GB/s |

---

## Troubleshooting

### Out of Memory (GPU)

```python
# Reduce batch size
BATCH_SIZE = 256  # Instead of 512

# Or use gradient accumulation
accumulation_steps = 4
```

### Slow Data Loading

```python
# Use more workers
DataLoader(..., num_workers=8, pin_memory=True)

# Or preload to RAM
df = pd.read_parquet('data.parquet')  # Loads entire file
```

### API Rate Limits

```python
# Automatic retry with exponential backoff
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=60))
async def fetch_data():
    ...
```

---

## Next Steps Checklist

- [ ] Setup API keys (`python scripts/setup_api_keys.py --interactive`)
- [ ] Verify GPU (`python scripts/verify_gpu_setup.py`)
- [ ] Collect historical data (`python scripts/collect_historical_data.py`)
- [ ] Verify data integrity (`python scripts/collect_historical_data.py --verify`)
- [ ] Run data analysis (`python scripts/analyze_data.py`)
- [ ] Train LSTM model (`python scripts/train_lstm.py`)
- [ ] Train PPO agent (`python scripts/train_ppo_agent.py`)
- [ ] Setup automated updates (Task Scheduler)
- [ ] Integrate with live trading

---

## Resources

- **GPU Optimization**: `RTX_5070Ti_ML_SETUP.md`
- **API Documentation**: `mcp_trader/data/multi_source_pipeline.py`
- **Data Collection**: `mcp_trader/data/historical_collector.py`
- **Analysis Tools**: `mcp_trader/analysis/data_analyzer.py`
- **Trading Models**: `mcp_trader/models/`

---

**You're now ready to build a world-class AI trading system! 🚀**



