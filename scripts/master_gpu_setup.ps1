#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Master orchestrator for RTX 5070 Ti GPU setup
.DESCRIPTION
    Guides through complete PyTorch build process with sm_120 support
    Estimated time: 2.5-3.5 hours
#>

$ErrorActionPreference = "Continue"

Write-Host @"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║          RTX 5070 Ti GPU Setup - Master Orchestrator           ║
║                                                                ║
║          Complete PyTorch Build with sm_120 Support            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

Write-Host "`nThis script will guide you through:" -ForegroundColor Yellow
Write-Host "  1. CUDA 12.8 installation (30-45 min)" -ForegroundColor Gray
Write-Host "  2. Build environment setup (15-20 min)" -ForegroundColor Gray
Write-Host "  3. PyTorch source build (60-90 min)" -ForegroundColor Gray
Write-Host "  4. GPU verification (10 min)" -ForegroundColor Gray
Write-Host "  5. AI model training (30-60 min)" -ForegroundColor Gray
Write-Host "`n  Total estimated time: 2.5-3.5 hours" -ForegroundColor Cyan

$startTime = Get-Date

# Create logs directory
$logDir = "D:\CodingFiles\AsterAI\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$masterLog = "$logDir\master_gpu_setup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Write-Host "`nMaster log: $masterLog" -ForegroundColor Gray

function Log-Step {
    param($message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $message"
    Write-Host $logMessage
    Add-Content -Path $masterLog -Value $logMessage
}

function Show-Menu {
    param($title, $options)
    Write-Host "`n" + "="*60 -ForegroundColor Cyan
    Write-Host $title -ForegroundColor Cyan
    Write-Host "="*60 -ForegroundColor Cyan
    for ($i = 0; $i -lt $options.Length; $i++) {
        Write-Host "  $($i+1). $($options[$i])" -ForegroundColor Gray
    }
}

# Welcome and prerequisites check
Write-Host "`n" + "="*60
Write-Host "Prerequisites Check" -ForegroundColor Cyan
Write-Host "="*60

Log-Step "Starting master GPU setup"

# Check GPU
Write-Host "`nChecking GPU..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    Write-Host "✅ GPU detected: $gpuInfo" -ForegroundColor Green
    Log-Step "GPU detected: $gpuInfo"
} catch {
    Write-Host "❌ GPU not detected! Please install NVIDIA drivers." -ForegroundColor Red
    Log-Step "ERROR: GPU not detected"
    exit 1
}

# Check CUDA
Write-Host "`nChecking CUDA..." -ForegroundColor Yellow
$cudaVersion = nvcc --version 2>$null | Select-String "release"
if ($cudaVersion) {
    Write-Host "Current CUDA: $cudaVersion" -ForegroundColor Gray
    if ($cudaVersion -match "12\.8") {
        Write-Host "✅ CUDA 12.8 already installed!" -ForegroundColor Green
        $cudaInstalled = $true
    } else {
        Write-Host "⚠️  CUDA 12.8 required (current: $cudaVersion)" -ForegroundColor Yellow
        $cudaInstalled = $false
    }
} else {
    Write-Host "⚠️  CUDA not detected" -ForegroundColor Yellow
    $cudaInstalled = $false
}

# Check Visual Studio
Write-Host "`nChecking Visual Studio 2022..." -ForegroundColor Yellow
if (Test-Path "C:\Program Files\Microsoft Visual Studio\2022") {
    Write-Host "✅ Visual Studio 2022 found" -ForegroundColor Green
    Log-Step "Visual Studio 2022 found"
} else {
    Write-Host "❌ Visual Studio 2022 not found!" -ForegroundColor Red
    Write-Host "Please install Visual Studio 2022 with C++ tools" -ForegroundColor Yellow
    Log-Step "ERROR: Visual Studio 2022 not found"
    exit 1
}

# Check Conda
Write-Host "`nChecking Conda..." -ForegroundColor Yellow
if (Get-Command conda -ErrorAction SilentlyContinue) {
    $condaVersion = conda --version
    Write-Host "✅ Conda found: $condaVersion" -ForegroundColor Green
    Log-Step "Conda found: $condaVersion"
} else {
    Write-Host "❌ Conda not found!" -ForegroundColor Red
    Write-Host "Please install Anaconda or Miniconda" -ForegroundColor Yellow
    Log-Step "ERROR: Conda not found"
    exit 1
}

# Check disk space
Write-Host "`nChecking disk space..." -ForegroundColor Yellow
$drive = Get-PSDrive D
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "Free space on D: $freeSpaceGB GB" -ForegroundColor Gray
if ($freeSpaceGB -lt 20) {
    Write-Host "⚠️  WARNING: Low disk space. Recommend at least 20 GB free." -ForegroundColor Yellow
    Log-Step "WARNING: Low disk space: $freeSpaceGB GB"
} else {
    Write-Host "✅ Sufficient disk space" -ForegroundColor Green
}

# Main menu
Write-Host "`n" + "="*60
Write-Host "Setup Options" -ForegroundColor Cyan
Write-Host "="*60

$options = @(
    "Complete automated setup (recommended)",
    "Step-by-step guided setup",
    "Install CUDA 12.8 only",
    "Setup build environment only",
    "Build PyTorch only (assumes CUDA 12.8 installed)",
    "Verify existing build",
    "View detailed guide",
    "Exit"
)

Show-Menu "What would you like to do?" $options

$choice = Read-Host "`nEnter choice (1-8)"

switch ($choice) {
    "1" {
        # Complete automated setup
        Log-Step "User selected: Complete automated setup"
        Write-Host "`nStarting complete automated setup..." -ForegroundColor Green
        
        if (-not $cudaInstalled) {
            Write-Host "`nPhase 1: CUDA 12.8 Installation" -ForegroundColor Cyan
            Log-Step "Starting CUDA 12.8 installation"
            & .\scripts\install_cuda_12.8.ps1
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "`n❌ CUDA installation failed or incomplete" -ForegroundColor Red
                Log-Step "ERROR: CUDA installation failed"
                Write-Host "Please install CUDA 12.8 manually and run this script again." -ForegroundColor Yellow
                exit 1
            }
        }
        
        Write-Host "`nPhase 2: Build Environment Setup" -ForegroundColor Cyan
        Log-Step "Starting build environment setup"
        & .\scripts\setup_pytorch_build.ps1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`n❌ Environment setup failed" -ForegroundColor Red
            Log-Step "ERROR: Environment setup failed"
            exit 1
        }
        
        Write-Host "`nPhase 3: PyTorch Build (This will take 60-90 minutes)" -ForegroundColor Cyan
        Log-Step "Starting PyTorch build"
        & .\scripts\build_pytorch_sm120.ps1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`n❌ PyTorch build failed" -ForegroundColor Red
            Log-Step "ERROR: PyTorch build failed"
            Write-Host "`nFallback option: Use CPU training" -ForegroundColor Yellow
            Write-Host "Run: python scripts\train_on_cpu.py" -ForegroundColor Cyan
            exit 1
        }
        
        Write-Host "`nPhase 4: Verification" -ForegroundColor Cyan
        Log-Step "Starting verification"
        conda activate pytorch_build
        python scripts\verify_gpu_build.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✅ SETUP COMPLETE!" -ForegroundColor Green
            Log-Step "Setup completed successfully"
            
            Write-Host "`nNext steps:" -ForegroundColor Cyan
            Write-Host "  1. Test LSTM: python scripts\test_lstm_gpu.py" -ForegroundColor Gray
            Write-Host "  2. Train AI: python scripts\quick_train_model.py" -ForegroundColor Gray
            Write-Host "  3. Deploy bot: python trading\ai_trading_bot.py" -ForegroundColor Gray
        } else {
            Write-Host "`n⚠️  Verification failed" -ForegroundColor Yellow
            Log-Step "WARNING: Verification failed"
        }
    }
    
    "2" {
        # Step-by-step guided setup
        Log-Step "User selected: Step-by-step guided setup"
        Write-Host "`nStep-by-step setup selected" -ForegroundColor Green
        Write-Host "Follow the prompts for each phase..." -ForegroundColor Gray
        
        # Guide through each step with pauses
        if (-not $cudaInstalled) {
            Write-Host "`n[Step 1/4] CUDA 12.8 Installation" -ForegroundColor Cyan
            $proceed = Read-Host "Proceed with CUDA installation? (y/n) [y]"
            if ($proceed -ne "n") {
                & .\scripts\install_cuda_12.8.ps1
            }
        }
        
        Write-Host "`n[Step 2/4] Build Environment Setup" -ForegroundColor Cyan
        $proceed = Read-Host "Proceed with environment setup? (y/n) [y]"
        if ($proceed -ne "n") {
            & .\scripts\setup_pytorch_build.ps1
        }
        
        Write-Host "`n[Step 3/4] PyTorch Build (60-90 minutes)" -ForegroundColor Cyan
        $proceed = Read-Host "Proceed with PyTorch build? (y/n) [y]"
        if ($proceed -ne "n") {
            & .\scripts\build_pytorch_sm120.ps1
        }
        
        Write-Host "`n[Step 4/4] Verification" -ForegroundColor Cyan
        $proceed = Read-Host "Proceed with verification? (y/n) [y]"
        if ($proceed -ne "n") {
            conda activate pytorch_build
            python scripts\verify_gpu_build.py
        }
    }
    
    "3" {
        # Install CUDA 12.8 only
        Log-Step "User selected: Install CUDA 12.8 only"
        & .\scripts\install_cuda_12.8.ps1
    }
    
    "4" {
        # Setup build environment only
        Log-Step "User selected: Setup build environment only"
        & .\scripts\setup_pytorch_build.ps1
    }
    
    "5" {
        # Build PyTorch only
        Log-Step "User selected: Build PyTorch only"
        & .\scripts\build_pytorch_sm120.ps1
    }
    
    "6" {
        # Verify existing build
        Log-Step "User selected: Verify existing build"
        Write-Host "`nVerifying existing PyTorch build..." -ForegroundColor Yellow
        conda activate pytorch_build
        python scripts\verify_gpu_build.py
        python scripts\test_lstm_gpu.py
    }
    
    "7" {
        # View detailed guide
        Log-Step "User selected: View detailed guide"
        Write-Host "`nOpening GPU_BUILD_GUIDE.md..." -ForegroundColor Yellow
        Start-Process "GPU_BUILD_GUIDE.md"
    }
    
    "8" {
        # Exit
        Log-Step "User exited"
        Write-Host "`nExiting..." -ForegroundColor Yellow
        exit 0
    }
    
    default {
        Write-Host "`nInvalid choice. Exiting..." -ForegroundColor Red
        exit 1
    }
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "`n" + "="*60
Write-Host "Master Setup Complete" -ForegroundColor Green
Write-Host "="*60

Write-Host "`nTotal time: $([math]::Round($duration.TotalMinutes, 1)) minutes" -ForegroundColor Cyan
Write-Host "Log file: $masterLog" -ForegroundColor Gray

Log-Step "Master setup completed in $([math]::Round($duration.TotalMinutes, 1)) minutes"

Write-Host "`nFor detailed documentation, see:" -ForegroundColor Yellow
Write-Host "  • GPU_BUILD_GUIDE.md - Complete build guide" -ForegroundColor Gray
Write-Host "  • GPU_BUILD_LOG.md - Your build log template" -ForegroundColor Gray
Write-Host "  • $masterLog - This session's log" -ForegroundColor Gray

