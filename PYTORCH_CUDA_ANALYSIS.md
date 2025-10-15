# PyTorch + CUDA Installation Analysis - First Principles

## 🔍 Current Situation Analysis

### What We Have
1. **Hardware**: RTX 5070 Ti (16GB VRAM) - ✅ Excellent
2. **Drivers**: NVIDIA 581.57 with CUDA 13.0 runtime - ✅ Working
3. **CUDA Toolkit**: v12.4 installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4` - ✅ Installed
4. **Python**: 3.13.3 (64-bit) - ✅ Working
5. **OS**: Windows 11 - ✅ Compatible

### The Core Problem
**PyTorch CUDA binary requires CUDA runtime DLLs that must match the compiled CUDA version**

Your system has:
- **CUDA Driver**: 13.0 (from GPU drivers - this is fine)
- **CUDA Toolkit**: 12.4 (manually installed - this is what PyTorch needs)
- **PyTorch needs**: CUDA 12.4 runtime DLLs accessible

## 🎯 Root Cause Analysis

### Why Installation Fails

1. **Path Issue**: CUDA 12.4 DLLs not in system PATH permanently
2. **Environment Isolation**: Virtual env doesn't inherit CUDA paths
3. **DLL Loading**: PyTorch looks for specific CUDA DLL files at runtime
4. **Version Mismatch**: PyTorch CUDA 12.1 incompatible with CUDA 12.4 toolkit

### Required DLL Files
PyTorch needs these DLLs from CUDA toolkit:
- `cublas64_12.dll`
- `cublasLt64_12.dll`
- `cudnn64_9.dll`
- `cusparse64_12.dll`
- `cufft64_11.dll`

## ✅ Clean Solution - Step by Step

### Approach 1: System-Wide Installation (Recommended)

**Pros:**
- Works across all projects
- Permanent solution
- Easier to manage

**Cons:**
- Modifies system PATH
- Affects all Python environments

### Approach 2: Per-Project Environment Variables

**Pros:**
- Project isolation
- No system changes

**Cons:**
- Must set env vars each time
- More complex

### Approach 3: Conda Environment (Best for ML)

**Pros:**
- Handles CUDA automatically
- Package management included
- Industry standard for ML

**Cons:**
- Requires Conda/Miniconda installation
- Larger download

## 🚀 Recommended Implementation Plan

### Step 1: System-Level CUDA Path Setup (One-time)

```powershell
# Add to USER PATH (persistent, no admin needed)
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
$cudaLibPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp"

if ($currentPath -notlike "*$cudaPath*") {
    [Environment]::SetEnvironmentVariable(
        "PATH",
        "$cudaPath;$cudaLibPath;$currentPath",
        "User"
    )
    echo "✅ CUDA paths added to USER PATH"
}

# Also set CUDA_HOME
[Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4", "User")
[Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4", "User")
```

### Step 2: Restart Terminal
**Critical**: Environment variables only load in new sessions

### Step 3: Verify CUDA is Accessible

```powershell
nvcc --version  # Should show CUDA 12.4
echo $env:CUDA_HOME  # Should show path
```

### Step 4: Install PyTorch CUDA

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 5: Test Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")

# Test GPU
device = torch.device("cuda")
x = torch.randn(1000, 1000).to(device)
y = torch.matmul(x, x.T)
print("✅ GPU tensor operations successful!")
```

## 🔧 Alternative: Conda Installation (Easier)

If above fails, use Conda (handles CUDA automatically):

### Step 1: Install Miniconda
Download: https://docs.conda.io/en/latest/miniconda.html

### Step 2: Create Conda Environment
```powershell
conda create -n asterai python=3.10 -y
conda activate asterai
```

### Step 3: Install PyTorch with CUDA
```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### Step 4: Test
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

This handles all CUDA dependencies automatically!

## 📋 Decision Matrix

| Method | Setup Time | Reliability | Flexibility | Recommendation |
|--------|------------|-------------|-------------|----------------|
| System PATH + pip | 5 min | Medium | High | Good for single GPU |
| Conda | 15 min | High | Medium | **Best for ML** |
| Docker GPU | 30 min | High | Low | Best for production |
| Manual DLL copy | 2 min | Low | Low | Not recommended |

## 🎯 My Recommendation

**Use Conda for your ML development**:

1. **Why Conda?**
   - Handles CUDA dependencies automatically
   - Works 99% of the time
   - Industry standard for data science
   - Easier environment management
   - Better package conflict resolution

2. **Quick Conda Setup**:
```powershell
# Install Miniconda (150MB)
# Download: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

# After installation:
conda create -n asterai python=3.10 -y
conda activate asterai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

3. **Why This Works**:
   - Conda downloads and manages CUDA libraries
   - No PATH manipulation needed
   - Isolated environment
   - Proven to work with RTX 40/50 series

## 🔥 Quick Fix for Current Setup

If you want to stick with pip (not recommended but possible):

```powershell
# Run this script to add CUDA to PATH permanently
python scripts/fix_cuda_path.py

# Restart terminal

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## ⚡ Immediate Action Plan

Choose ONE approach:

### Option A: Conda (Recommended - 15 minutes)
1. Download Miniconda
2. Install
3. Create environment
4. Install PyTorch via conda
5. Test → Should work immediately

### Option B: Fix Current Setup (5 minutes + restart)
1. Run PATH fix script
2. Restart terminal/IDE
3. Install PyTorch via pip
4. Test → Should work after restart

### Option C: Continue with CPU (0 minutes)
1. Use CPU PyTorch (already working)
2. Everything works, just slower
3. Add GPU support later

## 💡 Technical Details

### Why DLL Load Fails
```
ImportError: DLL load failed while importing _C: The specified module could not be found.
```

This means:
- PyTorch binary (`_C.pyd`) was compiled with CUDA 12.4
- At runtime, it tries to load CUDA DLLs
- Windows looks in: current directory, system PATH, PATH environment variable
- DLLs not found → import fails

### Solution
Make CUDA DLLs discoverable by adding CUDA\bin to PATH.

### Why Environment Variables Matter
PyTorch checks these at runtime:
1. `CUDA_HOME` - Where CUDA toolkit is installed
2. `CUDA_PATH` - Alternative to CUDA_HOME
3. `PATH` - Where to find DLL files

## 🎯 Final Recommendation

**Go with Conda**. It's designed for exactly this use case and handles all the complexity for you.

Would you like me to:
1. **Create Conda setup script** (automated)
2. **Create PATH fix script** (for pip approach)  
3. **Continue with CPU** (working now)

Your call! 🚀
