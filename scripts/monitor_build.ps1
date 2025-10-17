#!/usr/bin/env pwsh
# Monitor PyTorch build progress

Write-Host "Monitoring PyTorch Build Progress..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
Write-Host ""

while ($true) {
    Clear-Host
    Write-Host "="*60 -ForegroundColor Cyan
    Write-Host "PyTorch Build Monitor - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "="*60 -ForegroundColor Cyan
    
    # Check Python processes
    $pythonProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue
    $compilerProcs = Get-Process -Name "cl", "cmake", "ninja", "link" -ErrorAction SilentlyContinue
    
    Write-Host "`nActive Processes:" -ForegroundColor Yellow
    if ($pythonProcs) {
        Write-Host "  Python: $($pythonProcs.Count) processes" -ForegroundColor Green
    }
    if ($compilerProcs) {
        Write-Host "  Compilers: $($compilerProcs.Count) processes" -ForegroundColor Green
    }
    
    # GPU Status
    $gpuInfo = & "C:\Windows\System32\nvidia-smi.exe" --query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits
    $gpuData = $gpuInfo -split ','
    
    Write-Host "`nGPU Status:" -ForegroundColor Yellow
    Write-Host "  Utilization: $($gpuData[0].Trim())%" -ForegroundColor Gray
    Write-Host "  Temperature: $($gpuData[1].Trim())°C" -ForegroundColor Gray
    Write-Host "  Memory: $($gpuData[2].Trim()) MB / $($gpuData[3].Trim()) MB" -ForegroundColor Gray
    
    # Check for build directory
    if (Test-Path "D:\CodingFiles\pytorch\build") {
        $buildSize = (Get-ChildItem "D:\CodingFiles\pytorch\build" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host "`nBuild Progress:" -ForegroundColor Yellow
        Write-Host "  Build directory size: $([math]::Round($buildSize, 2)) MB" -ForegroundColor Gray
    }
    
    Write-Host "`nNext update in 30 seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 30
}
