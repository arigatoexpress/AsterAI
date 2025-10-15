# Fix CUDA PATH Permanently for AsterAI
# Run this script ONCE to add CUDA to your system PATH

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "AsterAI - Permanent CUDA PATH Configuration" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# CUDA installation path
$cudaBasePath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$cudaBinPath = "$cudaBasePath\bin"
$cudaLibPath = "$cudaBasePath\libnvvp"

# Check if CUDA exists
if (-not (Test-Path $cudaBasePath)) {
    Write-Host "[ERROR] CUDA 12.4 not found at: $cudaBasePath" -ForegroundColor Red
    Write-Host "Please install CUDA Toolkit 12.4 first" -ForegroundColor Yellow
    Write-Host "Download: https://developer.nvidia.com/cuda-12-4-0-download-archive" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] CUDA 12.4 found at: $cudaBasePath" -ForegroundColor Green
Write-Host ""

# Get current USER PATH
$currentUserPath = [Environment]::GetEnvironmentVariable("PATH", "User")

# Add CUDA paths if not present
$pathsToAdd = @($cudaBinPath, $cudaLibPath)
$pathModified = $false

foreach ($path in $pathsToAdd) {
    if ($currentUserPath -notlike "*$path*") {
        Write-Host "[ADDING] $path" -ForegroundColor Yellow
        $currentUserPath = "$path;$currentUserPath"
        $pathModified = $true
    } else {
        Write-Host "[SKIP] Already in PATH: $path" -ForegroundColor Gray
    }
}

# Update PATH if modified
if ($pathModified) {
    [Environment]::SetEnvironmentVariable("PATH", $currentUserPath, "User")
    Write-Host "[OK] PATH updated successfully" -ForegroundColor Green
} else {
    Write-Host "[OK] PATH already contains CUDA paths" -ForegroundColor Green
}

# Set CUDA_HOME and CUDA_PATH
[Environment]::SetEnvironmentVariable("CUDA_HOME", $cudaBasePath, "User")
[Environment]::SetEnvironmentVariable("CUDA_PATH", $cudaBasePath, "User")

Write-Host "[OK] CUDA_HOME set to: $cudaBasePath" -ForegroundColor Green
Write-Host "[OK] CUDA_PATH set to: $cudaBasePath" -ForegroundColor Green

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "CUDA Path Configuration Complete!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: You must restart your terminal/IDE for changes to take effect!" -ForegroundColor Yellow
Write-Host ""
Write-Host "After restarting:" -ForegroundColor Cyan
Write-Host "  1. Test CUDA: nvcc --version" -ForegroundColor White
Write-Host "  2. Install PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor White
Write-Host "  3. Test GPU: python -c \"import torch; print('CUDA:', torch.cuda.is_available())\"" -ForegroundColor White
Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan

