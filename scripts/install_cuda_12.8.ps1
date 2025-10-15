#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Download and install CUDA 12.8 toolkit for RTX 5070 Ti support
.DESCRIPTION
    Automated installer for CUDA 12.8 with cuDNN 9.x
    Required for PyTorch sm_120 (Blackwell) support
#>

Write-Host @"
╔════════════════════════════════════════════════════════════════╗
║              CUDA 12.8 Installation for RTX 5070 Ti            ║
║              Required for PyTorch sm_120 Support               ║
╚════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Check current CUDA version
Write-Host "`nChecking current CUDA installation..." -ForegroundColor Yellow
$currentCuda = & nvcc --version 2>$null
if ($currentCuda) {
    Write-Host "Current CUDA version detected:" -ForegroundColor Green
    Write-Host $currentCuda
} else {
    Write-Host "No CUDA installation detected" -ForegroundColor Yellow
}

# CUDA 12.8 download URL
$cudaUrl = "https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_560.76_windows.exe"
$cudaInstaller = "$env:TEMP\cuda_12.8.0_installer.exe"

Write-Host "`n" + "="*60
Write-Host "CUDA 12.8 Installation Steps:" -ForegroundColor Cyan
Write-Host "="*60

Write-Host @"

Option 1: MANUAL INSTALLATION (Recommended for first-time)
----------------------------------------------------------
1. Open browser and go to:
   https://developer.nvidia.com/cuda-downloads

2. Select:
   - Operating System: Windows
   - Architecture: x86_64
   - Version: 11
   - Installer Type: exe (local)

3. Download CUDA 12.8.0 (~3.5 GB)

4. Run installer with these options:
   ✓ CUDA Toolkit
   ✓ CUDA Samples
   ✓ CUDA Documentation
   ✓ Nsight Systems
   ✓ Nsight Compute
   ✓ Visual Studio Integration

5. Installation path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

6. Reboot after installation


Option 2: AUTOMATED DOWNLOAD (Advanced)
----------------------------------------
This script can attempt to download CUDA 12.8 automatically.
WARNING: Download is ~3.5 GB and may take 15-30 minutes.

"@

$choice = Read-Host "Choose installation method (1=Manual, 2=Automated) [1]"

if ($choice -eq "2") {
    Write-Host "`nStarting automated download..." -ForegroundColor Yellow
    Write-Host "Downloading CUDA 12.8.0 (~3.5 GB)..." -ForegroundColor Cyan
    Write-Host "This may take 15-30 minutes depending on your connection..." -ForegroundColor Yellow
    
    try {
        # Download with progress
        $ProgressPreference = 'Continue'
        Invoke-WebRequest -Uri $cudaUrl -OutFile $cudaInstaller -UseBasicParsing
        
        Write-Host "`n✅ Download complete!" -ForegroundColor Green
        Write-Host "Installer saved to: $cudaInstaller" -ForegroundColor Cyan
        
        $install = Read-Host "`nRun installer now? (y/n) [y]"
        if ($install -ne "n") {
            Write-Host "`nLaunching CUDA installer..." -ForegroundColor Yellow
            Write-Host "Follow the installation wizard." -ForegroundColor Cyan
            Start-Process -FilePath $cudaInstaller -Wait
            
            Write-Host "`n✅ CUDA installation complete!" -ForegroundColor Green
        }
    } catch {
        Write-Host "`n❌ Download failed: $_" -ForegroundColor Red
        Write-Host "Please use Manual Installation (Option 1)" -ForegroundColor Yellow
    }
} else {
    Write-Host "`nPlease complete manual installation and then continue." -ForegroundColor Yellow
    Write-Host "After installation, verify with: nvcc --version" -ForegroundColor Cyan
}

Write-Host "`n" + "="*60
Write-Host "Post-Installation Verification" -ForegroundColor Cyan
Write-Host "="*60

Write-Host @"

After CUDA 12.8 is installed, run these commands to verify:

1. Check CUDA version:
   nvcc --version

2. Check NVIDIA driver:
   nvidia-smi

3. Verify CUDA path:
   echo `$env:CUDA_PATH

Expected output:
- nvcc: release 12.8, V12.8.x
- nvidia-smi: Driver Version 560.76 or higher
- CUDA_PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

"@

$verify = Read-Host "Verify CUDA installation now? (y/n) [y]"
if ($verify -ne "n") {
    Write-Host "`nVerifying CUDA 12.8 installation..." -ForegroundColor Yellow
    
    # Check nvcc
    Write-Host "`n1. NVCC Version:" -ForegroundColor Cyan
    & nvcc --version
    
    # Check nvidia-smi
    Write-Host "`n2. NVIDIA Driver:" -ForegroundColor Cyan
    & nvidia-smi
    
    # Check CUDA_PATH
    Write-Host "`n3. CUDA Path:" -ForegroundColor Cyan
    if ($env:CUDA_PATH) {
        Write-Host "CUDA_PATH = $env:CUDA_PATH" -ForegroundColor Green
    } else {
        Write-Host "⚠️  CUDA_PATH not set. Setting now..." -ForegroundColor Yellow
        $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
        [Environment]::SetEnvironmentVariable("CUDA_PATH", $env:CUDA_PATH, "Machine")
        Write-Host "✅ CUDA_PATH set to: $env:CUDA_PATH" -ForegroundColor Green
    }
    
    Write-Host "`n✅ Verification complete!" -ForegroundColor Green
    Write-Host "You can now proceed to build PyTorch." -ForegroundColor Cyan
}

Write-Host "`n" + "="*60
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "="*60
Write-Host @"

1. Reboot your system (recommended)
2. Run: .\scripts\setup_pytorch_build.ps1
3. Run: .\scripts\build_pytorch_sm120.ps1

"@

