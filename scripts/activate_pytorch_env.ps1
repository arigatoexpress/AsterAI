# PyTorch Build Environment Variables
# Source this script before building PyTorch

$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:USE_CUDA = "1"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:CUDA_FORCE_PTX_JIT = "1"
$env:MAX_JOBS = "8"

Write-Host "âœ… PyTorch build environment activated" -ForegroundColor Green
