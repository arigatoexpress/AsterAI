# AsterAI GPU Setup Guide

## Current Status ✅

Your AsterAI system is **fully functional** with CPU fallback! We've successfully demonstrated:

- ✅ **Data Pipeline**: Multi-source data fetching (CoinGecko, Yahoo Finance, Alpha Vantage, FRED, etc.)
- ✅ **ML Training**: CPU-based machine learning with Random Forest models
- ✅ **Trading Simulation**: Strategy backtesting and performance metrics
- ✅ **GPU Detection**: Automatic GPU/CPU fallback logic
- ✅ **Dashboard**: Multi-page web interface with real-time monitoring
- ✅ **Data Analysis**: Visualization and statistical analysis tools

## GPU Setup Instructions

### Step 1: Verify Your Hardware
Your RTX 5070 Ti with CUDA 13.0 drivers is detected correctly:
```bash
nvidia-smi
```
Shows: Driver Version: 581.57, CUDA Version: 13.0

### Step 2: Install CUDA Toolkit 12.4
CUDA 13.0 PyTorch support is not yet available for Python 3.13.3. Use CUDA 12.4 (backward compatible):

**Option A: Manual Installation**
1. Download CUDA 12.4 Toolkit: https://developer.nvidia.com/cuda-12-4-0-download-archive
2. Run installer with custom options
3. Install CUDA Toolkit and Visual Studio Integration
4. Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`

**Option B: Automated Script**
```bash
python scripts/cuda_installer.py
```

### Step 3: Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Verify GPU Setup
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Step 5: Run GPU Training
```bash
python scripts/train_lstm_gpu.py
```

## Alternative GPU Setup Methods

### Method 2: Conda Installation
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### Method 3: Docker GPU
```bash
docker build -f Dockerfile.gpu -t asterai-gpu .
docker run --gpus all asterai-gpu
```

## Troubleshooting

### Issue: "DLL load failed"
- Ensure CUDA toolkit is properly installed
- Check PATH includes CUDA bin directory
- Try reinstalling PyTorch

### Issue: "No CUDA-capable device"
- Verify GPU drivers: `nvidia-smi`
- Check CUDA compatibility
- Update GPU drivers if needed

### Issue: Installation hangs
- Use `--no-cache-dir` flag
- Try different PyTorch index URL
- Check internet connection

## Performance Expectations

With RTX 5070 Ti (16GB VRAM):
- **LSTM Training**: ~10-50x faster than CPU
- **Batch Size**: 512-2048 samples
- **Memory Usage**: Monitor with `nvidia-smi`
- **Mixed Precision**: Use `torch.cuda.amp` for 2x speedup

## CPU Fallback Works!

Your system automatically detects and uses CPU when GPU is unavailable. All features work:
- Data collection and processing
- Model training and inference
- Strategy simulation
- Dashboard and monitoring

## Next Steps

1. **Immediate**: Use CPU version for development and testing
2. **GPU Setup**: Follow steps above when ready for accelerated training
3. **Scale Up**: Add multiple GPUs or cloud instances later
4. **Production**: Deploy with GPU-optimized containers

## Quick Start (CPU Mode)

```bash
# Install dependencies
pip install -r requirements.txt

# Test data pipeline
python scripts/test_data_pipeline.py

# Run CPU training demo
python scripts/demo_cpu_training.py

# Start dashboard
python dashboard/aster_trader_dashboard.py --port 8001

# Collect historical data
python scripts/collect_historical_data.py
```

Your AsterAI system is production-ready with CPU fallback! 🚀

