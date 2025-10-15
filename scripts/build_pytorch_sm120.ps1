#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build PyTorch from source with sm_120 (Blackwell) support
.DESCRIPTION
    Orchestrates the complete PyTorch build process for RTX 5070 Ti
    Expected time: 60-90 minutes
#>

Write-Host @"
╔════════════════════════════════════════════════════════════════╗
║          PyTorch Source Build for RTX 5070 Ti                  ║
║          Building with sm_120 (Blackwell) Support              ║
║          Expected Time: 60-90 minutes                          ║
╚════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

$startTime = Get-Date

# Check if setup was run
if (-not (conda env list | Select-String "pytorch_build")) {
    Write-Host "`n❌ Environment 'pytorch_build' not found!" -ForegroundColor Red
    Write-Host "Please run: .\scripts\setup_pytorch_build.ps1" -ForegroundColor Yellow
    exit 1
}

# Set environment variables
Write-Host "`nSetting build environment variables..." -ForegroundColor Yellow
. .\scripts\activate_pytorch_env.ps1

# Clone PyTorch if not already cloned
$pytorchDir = "D:\CodingFiles\pytorch"
if (-not (Test-Path $pytorchDir)) {
    Write-Host "`n" + "="*60
    Write-Host "Cloning PyTorch Repository" -ForegroundColor Cyan
    Write-Host "="*60
    
    Write-Host "`nCloning PyTorch from GitHub..." -ForegroundColor Yellow
    Write-Host "This will download ~2 GB and may take 10-15 minutes..." -ForegroundColor Gray
    
    Set-Location "D:\CodingFiles"
    git clone --recursive https://github.com/pytorch/pytorch
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n❌ Failed to clone PyTorch repository" -ForegroundColor Red
        exit 1
    }
    
    Set-Location $pytorchDir
    
    Write-Host "`nInitializing submodules..." -ForegroundColor Yellow
    git submodule sync
    git submodule update --init --recursive
    
    Write-Host "✅ PyTorch repository cloned" -ForegroundColor Green
} else {
    Write-Host "`n✅ PyTorch repository already exists at: $pytorchDir" -ForegroundColor Green
    Set-Location $pytorchDir
    
    # Update to latest
    $update = Read-Host "Update to latest PyTorch main branch? (y/n) [n]"
    if ($update -eq "y") {
        Write-Host "Updating repository..." -ForegroundColor Yellow
        git fetch origin
        git checkout main
        git pull origin main
        git submodule sync
        git submodule update --init --recursive
        Write-Host "✅ Repository updated" -ForegroundColor Green
    }
}

Write-Host "`n" + "="*60
Write-Host "Pre-Build Checklist" -ForegroundColor Cyan
Write-Host "="*60

Write-Host "`nVerifying build prerequisites..." -ForegroundColor Yellow

# Check CUDA
$cudaVersion = & nvcc --version | Select-String "release 12.8"
if ($cudaVersion) {
    Write-Host "✅ CUDA 12.8: OK" -ForegroundColor Green
} else {
    Write-Host "❌ CUDA 12.8: NOT FOUND" -ForegroundColor Red
    exit 1
}

# Check Python
$pythonVersion = python --version
Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green

# Check disk space
$drive = Get-PSDrive D
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "✅ Free disk space: $freeSpaceGB GB" -ForegroundColor Green
if ($freeSpaceGB -lt 20) {
    Write-Host "⚠️  WARNING: Low disk space. Recommend at least 20 GB free." -ForegroundColor Yellow
}

# Check RAM
$ram = Get-CimInstance Win32_ComputerSystem
$totalRAM = [math]::Round($ram.TotalPhysicalMemory / 1GB, 2)
Write-Host "✅ Total RAM: $totalRAM GB" -ForegroundColor Green
if ($totalRAM -lt 16) {
    Write-Host "⚠️  WARNING: Low RAM. Build may be slow. Consider reducing MAX_JOBS." -ForegroundColor Yellow
    $env:MAX_JOBS = "4"
}

Write-Host "`n" + "="*60
Write-Host "Starting PyTorch Build" -ForegroundColor Cyan
Write-Host "="*60

Write-Host @"

Build Configuration:
- CUDA Architecture: sm_120 (Blackwell)
- CUDA Version: 12.8
- Python: 3.13
- Parallel Jobs: $env:MAX_JOBS
- Build Type: Development (setup.py develop)

This will take 60-90 minutes. Progress will be shown below.
You can monitor GPU/CPU usage in Task Manager.

Build stages:
1. CMake configuration (~5 min)
2. C++ compilation (~50-70 min)
3. Python bindings (~10 min)
4. Installation (~5 min)

"@ -ForegroundColor Gray

$proceed = Read-Host "Start build now? (y/n) [y]"
if ($proceed -eq "n") {
    Write-Host "Build cancelled." -ForegroundColor Yellow
    exit 0
}

# Create build log directory
$logDir = "D:\CodingFiles\AsterAI\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$buildLog = "$logDir\pytorch_build_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Write-Host "`nBuild log will be saved to: $buildLog" -ForegroundColor Cyan

# Activate conda environment and build
Write-Host "`nActivating conda environment and starting build..." -ForegroundColor Yellow
Write-Host "="*60 -ForegroundColor Gray

# Create build script
$buildScript = @"
conda activate pytorch_build

# Set environment variables
`$env:TORCH_CUDA_ARCH_LIST = "12.0"
`$env:USE_CUDA = "1"
`$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
`$env:CUDA_HOME = `$env:CUDA_PATH
`$env:PATH = "`$env:CUDA_PATH\bin;`$env:PATH"
`$env:CUDA_FORCE_PTX_JIT = "1"
`$env:MAX_JOBS = "$env:MAX_JOBS"

# Change to PyTorch directory
Set-Location "$pytorchDir"

# Clean previous build (optional)
if (Test-Path "build") {
    Write-Host "Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build
}

# Start build
Write-Host "`nStarting PyTorch compilation..." -ForegroundColor Green
Write-Host "Time: `$(Get-Date)" -ForegroundColor Gray

python setup.py develop 2>&1 | Tee-Object -FilePath "$buildLog"

if (`$LASTEXITCODE -eq 0) {
    Write-Host "`n✅ BUILD SUCCESSFUL!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n❌ BUILD FAILED!" -ForegroundColor Red
    Write-Host "Check log file: $buildLog" -ForegroundColor Yellow
    exit 1
}
"@

$buildScriptPath = "$env:TEMP\pytorch_build.ps1"
$buildScript | Out-File -FilePath $buildScriptPath -Encoding UTF8

# Run build script
try {
    & powershell -ExecutionPolicy Bypass -File $buildScriptPath
    $buildSuccess = $LASTEXITCODE -eq 0
} catch {
    Write-Host "`n❌ Build error: $_" -ForegroundColor Red
    $buildSuccess = $false
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "`n" + "="*60
if ($buildSuccess) {
    Write-Host "BUILD COMPLETED SUCCESSFULLY!" -ForegroundColor Green
} else {
    Write-Host "BUILD FAILED" -ForegroundColor Red
}
Write-Host "="*60

Write-Host "`nBuild Statistics:" -ForegroundColor Cyan
Write-Host "  Start Time: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
Write-Host "  End Time: $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
Write-Host "  Duration: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor Gray
Write-Host "  Log File: $buildLog" -ForegroundColor Gray

if ($buildSuccess) {
    Write-Host "`n" + "="*60
    Write-Host "Next Steps" -ForegroundColor Cyan
    Write-Host "="*60
    
    Write-Host @"

1. Verify GPU support:
   conda activate pytorch_build
   python scripts\verify_gpu_build.py

2. Test LSTM on GPU:
   python scripts\test_lstm_gpu.py

3. Train AI model:
   python scripts\quick_train_model.py

"@ -ForegroundColor Gray
    
    $verify = Read-Host "Run verification now? (y/n) [y]"
    if ($verify -ne "n") {
        Write-Host "`nRunning verification..." -ForegroundColor Yellow
        conda activate pytorch_build
        python scripts\verify_gpu_build.py
    }
} else {
    Write-Host "`n" + "="*60
    Write-Host "Troubleshooting" -ForegroundColor Yellow
    Write-Host "="*60
    
    Write-Host @"

Common build errors and solutions:

1. "no kernel image available"
   - Ensure CUDA 12.8 is installed
   - Check TORCH_CUDA_ARCH_LIST="12.0"

2. "NVTX not found"
   - Reinstall CUDA with Nsight tools

3. "Out of memory"
   - Reduce MAX_JOBS to 4 or 2
   - Close other applications

4. "Link errors"
   - Ensure cuDNN 9.x is in PATH
   - Check Visual Studio installation

Check the build log for details:
$buildLog

Fallback option:
If build continues to fail, use CPU training:
    python scripts\train_on_cpu.py

"@ -ForegroundColor Gray
}

