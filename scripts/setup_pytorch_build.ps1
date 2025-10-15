#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup environment for PyTorch source build with sm_120 support
.DESCRIPTION
    Creates conda environment, installs dependencies, sets environment variables
#>

Write-Host @"
╔════════════════════════════════════════════════════════════════╗
║          PyTorch Build Environment Setup                       ║
║          RTX 5070 Ti (sm_120) Support                          ║
╚════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Check prerequisites
Write-Host "`nChecking prerequisites..." -ForegroundColor Yellow

# Check CUDA 12.8
Write-Host "`n1. Checking CUDA 12.8..." -ForegroundColor Cyan
$nvccOutput = & nvcc --version 2>$null | Select-String "release 12.8"
if ($nvccOutput) {
    Write-Host "✅ CUDA 12.8 detected" -ForegroundColor Green
} else {
    Write-Host "❌ CUDA 12.8 not found!" -ForegroundColor Red
    Write-Host "Please run: .\scripts\install_cuda_12.8.ps1" -ForegroundColor Yellow
    exit 1
}

# Check Visual Studio 2022
Write-Host "`n2. Checking Visual Studio 2022..." -ForegroundColor Cyan
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022"
if (Test-Path $vsPath) {
    Write-Host "✅ Visual Studio 2022 found" -ForegroundColor Green
    
    # Find vcvarsall.bat
    $vcvarsall = Get-ChildItem -Path $vsPath -Recurse -Filter "vcvarsall.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($vcvarsall) {
        Write-Host "   Location: $($vcvarsall.FullName)" -ForegroundColor Gray
    }
} else {
    Write-Host "❌ Visual Studio 2022 not found!" -ForegroundColor Red
    Write-Host "Please install Visual Studio 2022 with C++ tools" -ForegroundColor Yellow
    exit 1
}

# Check conda
Write-Host "`n3. Checking Conda..." -ForegroundColor Cyan
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if ($condaCmd) {
    Write-Host "✅ Conda found" -ForegroundColor Green
} else {
    Write-Host "❌ Conda not found!" -ForegroundColor Red
    Write-Host "Please install Anaconda or Miniconda" -ForegroundColor Yellow
    exit 1
}

# Check Git
Write-Host "`n4. Checking Git..." -ForegroundColor Cyan
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if ($gitCmd) {
    Write-Host "✅ Git found" -ForegroundColor Green
} else {
    Write-Host "❌ Git not found!" -ForegroundColor Red
    Write-Host "Please install Git for Windows" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n" + "="*60
Write-Host "Creating PyTorch Build Environment" -ForegroundColor Cyan
Write-Host "="*60

# Create conda environment
Write-Host "`nStep 1: Creating conda environment 'pytorch_build'..." -ForegroundColor Yellow
$envExists = conda env list | Select-String "pytorch_build"
if ($envExists) {
    Write-Host "Environment 'pytorch_build' already exists." -ForegroundColor Yellow
    $recreate = Read-Host "Recreate environment? (y/n) [n]"
    if ($recreate -eq "y") {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        conda env remove -n pytorch_build -y
        conda create -n pytorch_build python=3.13 -y
    }
} else {
    conda create -n pytorch_build python=3.13 -y
}

Write-Host "✅ Conda environment ready" -ForegroundColor Green

# Activate environment and install dependencies
Write-Host "`nStep 2: Installing build dependencies..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Gray

# Create a temporary script to run in the conda environment
$tempScript = @"
conda activate pytorch_build
conda install -y numpy mkl-devel ninja
pip install cmake typing_extensions pyyaml
pip install setuptools wheel
pip install requests
echo "✅ Dependencies installed"
"@

$tempScriptPath = "$env:TEMP\install_deps.ps1"
$tempScript | Out-File -FilePath $tempScriptPath -Encoding UTF8

# Run the script
& powershell -ExecutionPolicy Bypass -File $tempScriptPath

Write-Host "✅ Build dependencies installed" -ForegroundColor Green

# Download PyTorch requirements
Write-Host "`nStep 3: Downloading PyTorch requirements..." -ForegroundColor Yellow
try {
    $reqUrl = "https://raw.githubusercontent.com/pytorch/pytorch/main/requirements.txt"
    $reqFile = "$env:TEMP\pytorch_requirements.txt"
    Invoke-WebRequest -Uri $reqUrl -OutFile $reqFile -UseBasicParsing
    
    # Install PyTorch requirements
    $installReqScript = @"
conda activate pytorch_build
pip install -r $reqFile
echo "✅ PyTorch requirements installed"
"@
    $installReqScriptPath = "$env:TEMP\install_pytorch_req.ps1"
    $installReqScript | Out-File -FilePath $installReqScriptPath -Encoding UTF8
    & powershell -ExecutionPolicy Bypass -File $installReqScriptPath
    
    Write-Host "✅ PyTorch requirements installed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not download PyTorch requirements: $_" -ForegroundColor Yellow
    Write-Host "Will continue anyway..." -ForegroundColor Gray
}

Write-Host "`n" + "="*60
Write-Host "Environment Variables Configuration" -ForegroundColor Cyan
Write-Host "="*60

# Set environment variables for this session
Write-Host "`nSetting build environment variables..." -ForegroundColor Yellow

$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:USE_CUDA = "1"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:CUDA_FORCE_PTX_JIT = "1"
$env:MAX_JOBS = "8"
$env:CMAKE_PREFIX_PATH = "$env:CONDA_PREFIX"

Write-Host "✅ Environment variables set:" -ForegroundColor Green
Write-Host "   TORCH_CUDA_ARCH_LIST = $env:TORCH_CUDA_ARCH_LIST" -ForegroundColor Gray
Write-Host "   USE_CUDA = $env:USE_CUDA" -ForegroundColor Gray
Write-Host "   CUDA_PATH = $env:CUDA_PATH" -ForegroundColor Gray
Write-Host "   MAX_JOBS = $env:MAX_JOBS" -ForegroundColor Gray

# Create activation script for future use
Write-Host "`nCreating activation script for future builds..." -ForegroundColor Yellow
$activationScript = @"
# PyTorch Build Environment Variables
# Source this script before building PyTorch

`$env:TORCH_CUDA_ARCH_LIST = "12.0"
`$env:USE_CUDA = "1"
`$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
`$env:CUDA_HOME = `$env:CUDA_PATH
`$env:PATH = "`$env:CUDA_PATH\bin;`$env:PATH"
`$env:CUDA_FORCE_PTX_JIT = "1"
`$env:MAX_JOBS = "8"

Write-Host "✅ PyTorch build environment activated" -ForegroundColor Green
"@

$activationScript | Out-File -FilePath "scripts\activate_pytorch_env.ps1" -Encoding UTF8
Write-Host "✅ Activation script saved to: scripts\activate_pytorch_env.ps1" -ForegroundColor Green

Write-Host "`n" + "="*60
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "="*60

Write-Host @"

Environment 'pytorch_build' is ready for PyTorch compilation.

To activate the environment:
    conda activate pytorch_build

To set environment variables:
    .\scripts\activate_pytorch_env.ps1

Next steps:
    1. Run: .\scripts\build_pytorch_sm120.ps1
    2. Wait 60-90 minutes for compilation
    3. Run: python scripts\verify_gpu_build.py

"@ -ForegroundColor Cyan

# Ask if user wants to continue to build
$continue = Read-Host "`nProceed to PyTorch build now? (y/n) [y]"
if ($continue -ne "n") {
    Write-Host "`nStarting PyTorch build..." -ForegroundColor Yellow
    & .\scripts\build_pytorch_sm120.ps1
}

