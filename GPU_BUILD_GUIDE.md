# 🔧 Complete GPU Build Guide - RTX 5070 Ti PyTorch Setup

## Overview

This guide walks you through building PyTorch from source with sm_120 (Blackwell) support for your RTX 5070 Ti. Based on Grok 4's research and community testing.

**Total Time:** 2.5-3.5 hours  
**Difficulty:** Intermediate  
**Success Rate:** High (with CUDA 12.8)

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] **RTX 5070 Ti** installed and detected (`nvidia-smi` works)
- [ ] **Windows 11** (Build 26200 or later)
- [ ] **NVIDIA Drivers** 560.76+ installed
- [ ] **Visual Studio 2022** with C++ tools
- [ ] **Anaconda or Miniconda** installed
- [ ] **Git for Windows** installed
- [ ] **20+ GB free disk space** on D: drive
- [ ] **16+ GB RAM** (32 GB recommended)
- [ ] **Stable internet connection** (will download ~5 GB)

---

## Phase 1: Install CUDA 12.8 (30-45 minutes)

### Why CUDA 12.8?

CUDA 12.8 is the **minimum version** that supports sm_120 (Blackwell architecture). Your current CUDA 12.4 does NOT support it.

### Installation Steps

**Option A: Automated (Recommended)**

```powershell
.\scripts\install_cuda_12.8.ps1
```

Follow the prompts. The script will guide you through download and installation.

**Option B: Manual**

1. Go to: https://developer.nvidia.com/cuda-downloads
2. Select:
   - Operating System: **Windows**
   - Architecture: **x86_64**
   - Version: **11**
   - Installer Type: **exe (local)**
3. Download CUDA 12.8.0 (~3.5 GB)
4. Run installer
5. Select these components:
   - ✅ CUDA Toolkit
   - ✅ CUDA Samples
   - ✅ CUDA Documentation
   - ✅ Nsight Systems
   - ✅ Nsight Compute
   - ✅ Visual Studio Integration
6. Install to: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
7. **Reboot** after installation

### Verification

```powershell
nvcc --version
# Should show: release 12.8, V12.8.x

nvidia-smi
# Should show: Driver Version 560.76+

echo $env:CUDA_PATH
# Should show: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
```

---

## Phase 2: Setup Build Environment (15-20 minutes)

### Automated Setup

```powershell
.\scripts\setup_pytorch_build.ps1
```

This script will:
1. Create conda environment `pytorch_build`
2. Install build dependencies (numpy, mkl-devel, ninja, cmake)
3. Download PyTorch requirements
4. Set environment variables
5. Create activation script

### Manual Setup (if automated fails)

```powershell
# Create environment
conda create -n pytorch_build python=3.13 -y
conda activate pytorch_build

# Install dependencies
conda install -y numpy mkl-devel ninja
pip install cmake typing_extensions pyyaml setuptools wheel

# Download PyTorch requirements
$reqUrl = "https://raw.githubusercontent.com/pytorch/pytorch/main/requirements.txt"
Invoke-WebRequest -Uri $reqUrl -OutFile pytorch_requirements.txt
pip install -r pytorch_requirements.txt

# Set environment variables
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:USE_CUDA = "1"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:CUDA_FORCE_PTX_JIT = "1"
$env:MAX_JOBS = "8"
```

### Verification

```powershell
conda env list
# Should show: pytorch_build

python --version
# Should show: Python 3.13.x

cmake --version
# Should show: cmake version 3.x
```

---

## Phase 3: Clone PyTorch Repository (15-20 minutes)

### Automated (Included in build script)

The build script will clone PyTorch automatically if not present.

### Manual Clone

```powershell
cd D:\CodingFiles
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout main
git submodule sync
git submodule update --init --recursive
```

**Note:** This downloads ~2 GB and may take 10-15 minutes depending on your connection.

---

## Phase 4: Build PyTorch (60-90 minutes) ⏰

### The Main Event

```powershell
.\scripts\build_pytorch_sm120.ps1
```

This is the longest step. The script will:
1. Verify prerequisites
2. Set build environment variables
3. Run `python setup.py develop`
4. Log everything to `logs/pytorch_build_TIMESTAMP.log`

### What to Expect

**Build Stages:**
1. **CMake Configuration** (~5 min)
   - Detects CUDA, cuDNN, Visual Studio
   - Configures build system
   
2. **C++ Compilation** (~50-70 min)
   - Compiles PyTorch core
   - Compiles CUDA kernels for sm_120
   - Most CPU-intensive phase
   
3. **Python Bindings** (~10 min)
   - Builds Python extensions
   - Links everything together
   
4. **Installation** (~5 min)
   - Installs to conda environment
   - Creates egg-link

**Progress Indicators:**
- You'll see lots of `[X/Y]` compilation progress
- CPU usage will be 80-100%
- Fans may spin up (normal)
- No GPU usage yet (compilation is CPU-only)

### Monitoring Build

**Terminal 1:** Run build script  
**Terminal 2:** Monitor resources
```powershell
# Watch CPU/RAM usage
Get-Process python | Select-Object CPU, WorkingSet

# Check build log in real-time
Get-Content logs\pytorch_build_*.log -Wait -Tail 50
```

### Common Build Errors

**Error: "no kernel image available"**
- **Cause:** CUDA 12.8 not installed or not in PATH
- **Fix:** Verify `nvcc --version` shows 12.8

**Error: "NVTX not found"**
- **Cause:** CUDA installed without Nsight tools
- **Fix:** Reinstall CUDA with all components

**Error: "Out of memory"**
- **Cause:** Insufficient RAM
- **Fix:** Close other apps, reduce MAX_JOBS to 4 or 2

**Error: "Link error: cannot find cuDNN"**
- **Cause:** cuDNN not in PATH
- **Fix:** Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin` to PATH

**Error: "cl.exe not found"**
- **Cause:** Visual Studio C++ compiler not activated
- **Fix:** Run `vcvarsall.bat x64` first

### If Build Fails

1. Check the log file: `logs/pytorch_build_TIMESTAMP.log`
2. Search for the first ERROR message
3. Try solutions above
4. If still failing, use CPU training fallback:
   ```powershell
   python scripts\train_on_cpu.py
   ```

---

## Phase 5: Verification (10 minutes)

### Test 1: Basic GPU Support

```powershell
conda activate pytorch_build
python scripts\verify_gpu_build.py
```

**Expected Output:**
```
✅ SUCCESS: sm_120 support enabled!
✅ GPU tensor operations working!
```

### Test 2: LSTM on GPU

```powershell
python scripts\test_lstm_gpu.py
```

**Expected Output:**
```
✅ Forward pass successful!
✅ Backward pass successful!
✅ LSTM training on GPU works!
```

### Test 3: Quick GPU Check

```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_capability(0))  # (12, 0)
```

---

## Phase 6: Train AI Model (30-60 minutes)

### Run GPU-Accelerated Training

```powershell
conda activate pytorch_build
python scripts\quick_train_model.py
```

**Expected Performance:**
- Training time: **30-60 minutes** (vs 2-3 hours CPU)
- GPU utilization: **80-95%**
- Memory usage: **8-12 GB VRAM**
- Model accuracy: **65-75%**

### Monitor Training

**Terminal 1:** Training script  
**Terminal 2:** GPU monitoring
```powershell
nvidia-smi -l 1
```

Watch for:
- GPU utilization: Should be 80-95%
- Memory usage: Should increase during training
- Temperature: Should be 60-80°C (normal)
- Power: Should be near max TDP

---

## Phase 7: Deploy AI Trading Bot

### Update Trading Bot

Once training completes:

```powershell
python trading\ai_trading_bot.py
```

The bot will:
1. Load GPU-trained model
2. Run inference on CPU (fast enough)
3. Generate trading signals
4. Execute paper trades

---

## Troubleshooting Guide

### Issue: Build takes >2 hours

**Solutions:**
- Check CPU usage (should be 80-100%)
- Reduce MAX_JOBS if RAM is low
- Close other applications
- Check disk I/O (SSD recommended)

### Issue: "CUDA out of memory" during training

**Solutions:**
- Reduce batch size in training script
- Close other GPU applications
- Use gradient checkpointing
- Enable mixed precision (FP16)

### Issue: GPU utilization <50%

**Solutions:**
- Increase batch size
- Check data loading bottleneck
- Use pin_memory=True in DataLoader
- Increase num_workers in DataLoader

### Issue: Model accuracy <60%

**Solutions:**
- Train for more epochs
- Adjust learning rate
- Add more features
- Increase model size
- Check data quality

---

## Performance Optimization

### Mixed Precision Training (FP16)

Enable 2x speedup:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Checkpointing

Save memory for larger models:

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```

### DataLoader Optimization

```python
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

---

## Maintenance

### Update PyTorch

When official PyTorch 2.8+ releases (est. Dec 2025):

```powershell
conda activate pytorch_build
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Rebuild PyTorch

If you need to rebuild:

```powershell
cd D:\CodingFiles\pytorch
git pull origin main
git submodule update --init --recursive
python setup.py develop
```

### Clean Build

For a fresh start:

```powershell
cd D:\CodingFiles\pytorch
Remove-Item -Recurse -Force build
python setup.py develop
```

---

## Success Criteria

You've successfully completed the build when:

- [x] `torch.cuda.is_available()` returns `True`
- [x] `torch.cuda.get_device_capability(0)` returns `(12, 0)`
- [x] LSTM model trains on GPU without errors
- [x] Training time <1 hour for your dataset
- [x] GPU utilization 80-95% during training
- [x] Model accuracy 65-75%

---

## Fallback Plan

If PyTorch build fails after 2 hours:

### Option 1: CPU Training (Immediate)

```powershell
python scripts\train_on_cpu.py
```

- Time: 2-3 hours
- Accuracy: 60-65%
- Deploy tomorrow

### Option 2: Cloud GPU (Alternative)

Use Google Colab or Kaggle:
1. Upload your data
2. Train on T4/P100 GPU
3. Download trained model
4. Deploy locally

### Option 3: WSL2 (Linux)

Try building in WSL2:
1. Install WSL2 with Ubuntu
2. Install CUDA in WSL2
3. Build PyTorch in Linux
4. May have better compatibility

---

## Resources

- **PyTorch Build Guide:** https://github.com/pytorch/pytorch#from-source
- **CUDA Downloads:** https://developer.nvidia.com/cuda-downloads
- **NVIDIA Forums:** https://forums.developer.nvidia.com/
- **PyTorch Forums:** https://discuss.pytorch.org/
- **This Project:** Check `GPU_BUILD_LOG.md` for your build logs

---

## Support

If you encounter issues:

1. Check `logs/pytorch_build_*.log`
2. Review error messages
3. Try solutions in Troubleshooting section
4. Use CPU training as fallback
5. Document issues in `GPU_BUILD_LOG.md`

---

**Good luck with your build!** 🚀

The RTX 5070 Ti is powerful hardware - once PyTorch is built, you'll have blazing-fast AI training for your trading system.

