# Local Development Setup

Step-by-step guide to set up Rari Trade on your local machine for development and testing.

## System Requirements Check

### Python Version
```bash
python --version
# Should show: Python 3.11.x or higher
```

### Available RAM
```bash
# Windows
wmic ComputerSystem get TotalPhysicalMemory

# macOS
system_profiler SPHardwareDataType | grep Memory

# Linux
free -h
```
**Minimum**: 8GB RAM, **Recommended**: 16GB+

### Disk Space
```bash
# Check available space
# Windows: Right-click drive → Properties
# macOS/Linux: df -h
```
**Required**: 10GB free space

## Step 1: Repository Setup

### Clone Repository
```bash
git clone https://github.com/yourusername/rari-trade.git
cd rari-trade
```

### Verify Files
```bash
ls -la
# Should see: mcp_trader/, scripts/, config/, requirements.txt, etc.
```

## Step 2: Python Environment

### Create Virtual Environment
```bash
# Windows
python -m venv rari_trade_env
rari_trade_env\Scripts\activate

# macOS/Linux
python -m venv rari_trade_env
source rari_trade_env/bin/activate
```

### Verify Activation
```bash
which python
# Should point to: rari_trade_env/bin/python (or Scripts\python on Windows)
```

## Step 3: Dependencies Installation

### Core Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- pandas, numpy: Data processing
- torch: Machine learning
- ccxt: Exchange connectivity
- fastapi, uvicorn: Web API
- streamlit: Dashboard
- pytest: Testing framework

### GPU Support (Optional)
```bash
# Check CUDA availability
nvidia-smi

# If CUDA available, install GPU packages
pip install -r requirements-gpu.txt
```

### Verify Installation
```bash
python -c "import pandas, torch, ccxt; print('✓ Dependencies installed')"
```

## Step 4: Configuration Setup

### API Keys Configuration
```bash
cp config/api_keys_template.json config/api_keys.json
```

Edit `config/api_keys.json`:
```json
{
  "exchange_credentials": {
    "api_key": "your_exchange_api_key",
    "api_secret": "your_exchange_api_secret",
    "testnet": true
  },
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "rari_trade",
    "username": "rari_user",
    "password": "secure_password"
  }
}
```

### Trading Parameters
```bash
cp config/trading_config_template.json config/trading_config.json
```

## Step 5: Data Pipeline Setup

### Download Historical Data
```bash
python scripts/collect_historical_data.py
```

**Expected output:**
```
2024-01-15 10:30:00 - INFO - Collecting data for BTC/USDT
2024-01-15 10:30:05 - INFO - Downloaded 10000 candles
2024-01-15 10:30:10 - INFO - Calculating technical indicators
2024-01-15 10:30:15 - INFO - Data saved to data/historical/
```

### Verify Data Files
```bash
ls -la data/historical/
# Should see: BTCUSDT_1h.parquet, ETHUSDT_1h.parquet, etc.
```

## Step 6: Model Training

### CPU Training (Default)
```bash
python scripts/train_cpu_fallback.py
```

**Training process:**
1. Load historical data
2. Calculate 41 technical indicators
3. Split data: 70% training, 20% validation, 10% testing
4. Train 7 AI models
5. Validate performance
6. Save models to `models/` directory

**Expected duration:** 10-30 minutes

### GPU Training (If Available)
```bash
python scripts/train_gpu_accelerated.py
```

### Verify Models
```bash
ls -la models/
# Should see: ensemble_model.pkl, ppo_model.zip, etc.
```

## Step 7: Validation Testing

### Paper Trading Setup
```bash
python scripts/setup_paper_trading.py
```

**What it does:**
- Runs 7-day simulated trading
- Tests all risk management rules
- Validates performance metrics
- Generates detailed report

**Expected output:**
```
Paper Trading Validation Report
==============================
Period: 2024-01-01 to 2024-01-08
Total Trades: 156
Win Rate: 62.8%
Sharpe Ratio: 1.85
Max Drawdown: 7.2%
Total Return: 18.4%

✓ All validation criteria met
```

## Step 8: Dashboard Setup

### Start Local Dashboard
```bash
cd dashboard
python -m streamlit run unified_trading_dashboard.py --server.port 8501
```

### Access Dashboard
Open browser to: `http://localhost:8501`

**Dashboard features:**
- Portfolio overview
- Active positions
- Performance charts
- Risk metrics
- System status

## Step 9: Testing Suite

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Integration Tests
```bash
pytest tests/ -k "integration" -v
```

### Performance Testing
```bash
pytest tests/ -k "performance" --durations=10
```

## Step 10: Live Trading Preparation

### Pre-Launch Checklist
- [ ] Paper trading validation passed
- [ ] API keys configured correctly
- [ ] Risk parameters set appropriately
- [ ] Emergency stop procedures tested
- [ ] Backup systems ready

### Start Live Trading
```bash
python scripts/deploy_live_trading.py
```

**Initial deployment:**
- Starts with $100 (configurable)
- Conservative position sizing
- Full risk management active
- Real-time monitoring enabled

## Troubleshooting Common Issues

### Issue: Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: CUDA Not Available
```bash
# Check GPU status
nvidia-smi

# Fall back to CPU
export CUDA_VISIBLE_DEVICES=""
python scripts/train_cpu_fallback.py
```

### Issue: Memory Errors
```bash
# Reduce batch size in config
# Add swap file or increase RAM
# Use smaller dataset for testing
```

### Issue: API Connection Failed
```bash
# Check internet connection
# Verify API keys
# Test with different exchange
python scripts/test_api_connectivity.py
```

### Issue: Dashboard Not Loading
```bash
# Check port availability
netstat -an | grep 8501

# Kill existing processes
pkill -f streamlit

# Restart dashboard
python -m streamlit run unified_trading_dashboard.py
```

## Performance Optimization

### CPU Optimization
```bash
# Use multiple cores
export OMP_NUM_THREADS=8

# Optimize pandas
export PANDAS_OPTIMIZE=1
```

### Memory Optimization
```bash
# Use memory-efficient data types
# Process data in chunks
# Clear unused variables
```

### Storage Optimization
```bash
# Use compressed file formats (.parquet)
# Clean old log files regularly
# Archive historical data
```

## Monitoring & Logging

### Log Files Location
```
logs/
├── trading.log          # Main trading activity
├── system.log           # System health
├── errors.log           # Error tracking
└── performance.log      # Performance metrics
```

### Log Rotation
```bash
# Logs automatically rotate daily
# Old logs compressed and archived
# Maximum 30 days retention
```

## Backup & Recovery

### Data Backup
```bash
# Models backup
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Configuration backup
cp config/ config_backup_$(date +%Y%m%d)/ -r
```

### Recovery Procedure
1. Stop all running processes
2. Restore from backup
3. Verify data integrity
4. Restart services
5. Run validation tests

## Next Steps

### Development Workflow
1. Make code changes
2. Run tests: `pytest tests/`
3. Validate: `python scripts/setup_paper_trading.py`
4. Deploy: `python scripts/deploy_live_trading.py`

### Scaling Up
- Move to cloud deployment
- Add monitoring services
- Implement automated backups
- Set up alerting systems

---

*Your local Rari Trade setup is now complete and ready for development or live trading.*
