# 🚀 Quick Start - RTX 5070 Ti GPU Setup

## TL;DR - Get GPU Working in 3 Commands

```powershell
# 1. Run master setup (will guide you through everything)
.\scripts\master_gpu_setup.ps1

# 2. After build completes, verify
conda activate pytorch_build
python scripts\verify_gpu_build.py

# 3. Train AI model
python scripts\quick_train_model.py
```

**Total time:** 2.5-3.5 hours (mostly automated)

---

## What This Does

1. **Installs CUDA 12.8** (required for sm_120 support)
2. **Sets up build environment** (conda, dependencies)
3. **Builds PyTorch from source** (with Blackwell support)
4. **Verifies GPU works** (tests sm_120)
5. **Trains AI model** (GPU-accelerated)

---

## Current Status

✅ **What's Working:**
- Paper trading bot (running on CPU)
- Data collection (100 cryptos collected)
- Baseline strategy (60% win rate)

❌ **What's Blocked:**
- GPU training (RTX 5070 Ti too new for current PyTorch)

🎯 **Goal:**
- Build PyTorch with sm_120 support
- Train AI model on GPU (3-6x faster)
- Deploy AI-powered trading bot

---

## Prerequisites

Before running the setup:

- [ ] RTX 5070 Ti installed (`nvidia-smi` works)
- [ ] Windows 11
- [ ] Visual Studio 2022 with C++ tools
- [ ] Anaconda/Miniconda installed
- [ ] 20+ GB free disk space
- [ ] 2-3 hours of time

---

## Step-by-Step

### Option 1: Automated (Recommended)

```powershell
.\scripts\master_gpu_setup.ps1
```

Select option **1** (Complete automated setup) and let it run.

### Option 2: Manual

```powershell
# Step 1: Install CUDA 12.8
.\scripts\install_cuda_12.8.ps1

# Step 2: Setup environment
.\scripts\setup_pytorch_build.ps1

# Step 3: Build PyTorch (60-90 min)
.\scripts\build_pytorch_sm120.ps1

# Step 4: Verify
conda activate pytorch_build
python scripts\verify_gpu_build.py
python scripts\test_lstm_gpu.py

# Step 5: Train
python scripts\quick_train_model.py
```

---

## Timeline

| Phase | Duration | What Happens |
|-------|----------|--------------|
| CUDA 12.8 Install | 30-45 min | Download + install CUDA toolkit |
| Environment Setup | 15-20 min | Create conda env, install deps |
| PyTorch Clone | 15-20 min | Download PyTorch source (~2 GB) |
| PyTorch Build | 60-90 min | Compile with sm_120 support |
| Verification | 10 min | Test GPU, LSTM, tensor ops |
| AI Training | 30-60 min | Train model on GPU |
| **TOTAL** | **2.5-3.5 hours** | Mostly automated |

---

## What to Expect

### During Build (60-90 min)

- CPU usage: 80-100% (normal)
- Fans spinning up (normal)
- Lots of compilation messages
- Progress: `[X/Y]` counters
- No GPU usage yet (compilation is CPU-only)

### After Build

- `torch.cuda.is_available()` → `True`
- `torch.cuda.get_device_capability(0)` → `(12, 0)`
- LSTM trains on GPU without errors
- Training time: <1 hour (vs 3 hours CPU)

---

## Troubleshooting

### Build fails?

1. Check log: `logs/pytorch_build_*.log`
2. Try solutions in `GPU_BUILD_GUIDE.md`
3. Use fallback: `python scripts\train_on_cpu.py`

### CUDA 12.8 not installing?

1. Download manually from NVIDIA
2. Install with all components
3. Reboot
4. Verify: `nvcc --version`

### Out of memory during build?

1. Close other applications
2. Reduce MAX_JOBS to 4 or 2
3. Try again

---

## Success Criteria

You're done when:

- ✅ `torch.cuda.is_available()` = True
- ✅ `torch.cuda.get_device_capability(0)` = (12, 0)
- ✅ LSTM trains on GPU
- ✅ Training time <1 hour
- ✅ GPU utilization 80-95%

---

## Fallback Plan

If build fails after 2 hours:

**Option A: CPU Training**
```powershell
python scripts\train_on_cpu.py
```
Run overnight, deploy tomorrow.

**Option B: Cloud GPU**
Use Google Colab or Kaggle for training.

**Option C: WSL2**
Try building in Linux (may be easier).

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/master_gpu_setup.ps1` | Main orchestrator |
| `scripts/install_cuda_12.8.ps1` | CUDA installer |
| `scripts/setup_pytorch_build.ps1` | Environment setup |
| `scripts/build_pytorch_sm120.ps1` | PyTorch build |
| `scripts/verify_gpu_build.py` | GPU verification |
| `scripts/test_lstm_gpu.py` | LSTM test |
| `GPU_BUILD_GUIDE.md` | Detailed guide |
| `GPU_BUILD_LOG.md` | Build log template |

---

## Next Steps After GPU Works

1. **Test LSTM:** `python scripts/test_lstm_gpu.py`
2. **Train AI:** `python scripts/quick_train_model.py`
3. **Deploy bot:** `python trading/ai_trading_bot.py`
4. **Monitor:** `nvidia-smi -l 1`
5. **Compare:** GPU vs CPU performance

---

## Need Help?

1. Read: `GPU_BUILD_GUIDE.md` (comprehensive guide)
2. Check: `logs/pytorch_build_*.log` (build logs)
3. Review: `GPU_BUILD_LOG.md` (document your build)
4. Fallback: CPU training (always works)

---

## Ready to Start?

```powershell
.\scripts\master_gpu_setup.ps1
```

Choose option **1** and let it run!

**Good luck!** 🚀 Your RTX 5070 Ti is about to become a trading powerhouse.

