"""
Install all dependencies for AsterAI Trading System
Handles Windows-specific issues and provides clear feedback
"""

import subprocess
import sys
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════╗
║        AsterAI - Dependency Installation                       ║
╚════════════════════════════════════════════════════════════════╝
""")

def run_pip_install(package, description=""):
    """Install a package with pip"""
    print(f"\n{'='*70}")
    print(f"Installing: {package}")
    if description:
        print(f"Purpose: {description}")
    print(f"{'='*70}")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])
        print(f"[OK] {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install {package}: {e}")
        return False

# Core dependencies
print("\n[1/5] Installing Core Dependencies...")
core_packages = [
    ("numpy>=1.23.2,<2.0", "Numerical computing"),
    ("pandas>=2.0.0", "Data analysis"),
    ("requests>=2.31.0", "HTTP requests"),
    ("websockets>=12.0", "WebSocket support"),
    ("pydantic>=2.5.0", "Data validation"),
    ("pydantic-settings>=2.1.0", "Settings management"),
    ("python-dotenv>=1.0.0", "Environment variables"),
]

for package, desc in core_packages:
    run_pip_install(package, desc)

# Async and HTTP
print("\n[2/5] Installing Async & HTTP Libraries...")
async_packages = [
    ("aiohttp>=3.9.0", "Async HTTP"),
    ("aiodns>=3.1.0", "Async DNS"),
    ("yfinance>=0.2.30", "Yahoo Finance"),
]

for package, desc in async_packages:
    run_pip_install(package, desc)

# Dashboard
print("\n[3/5] Installing Dashboard & Visualization...")
dashboard_packages = [
    ("streamlit>=1.28.0", "Dashboard framework"),
    ("plotly>=5.17.0", "Interactive plots"),
    ("fastapi>=0.104.0", "Web API"),
    ("uvicorn[standard]>=0.24.0", "ASGI server"),
    ("jinja2>=3.1.2", "Template engine"),
    ("matplotlib>=3.7.0", "Plotting"),
    ("seaborn>=0.13.0", "Statistical visualization"),
]

for package, desc in dashboard_packages:
    run_pip_install(package, desc)

# Data & ML
print("\n[4/5] Installing Data & ML Libraries...")
ml_packages = [
    ("scikit-learn>=1.3.0", "Machine learning"),
    ("scipy>=1.11.0", "Scientific computing"),
    ("statsmodels>=0.14.0", "Statistical models"),
    ("arch>=6.0.0", "Time series"),
    ("pyarrow>=14.0.0", "Parquet format"),
    ("fastparquet>=2023.10.0", "Parquet support"),
]

for package, desc in ml_packages:
    run_pip_install(package, desc)

# Technical Analysis & Trading
print("\n[5/5] Installing Trading & TA Libraries...")
trading_packages = [
    ("pandas-ta>=0.3.14b0", "Technical analysis"),
    ("ta>=0.11.0", "TA indicators"),
    ("tenacity>=8.2.0", "Retry logic"),
    ("ratelimit>=2.2.1", "Rate limiting"),
    ("cryptography>=41.0.0", "Security"),
    ("web3>=6.0.0", "Blockchain"),
]

for package, desc in trading_packages:
    run_pip_install(package, desc)

# Optional: Deep Learning (only if CUDA is available)
print("\n" + "="*70)
print("GPU/Deep Learning Setup")
print("="*70)

try:
    import torch
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available:
        print("\n[WARNING] CUDA not detected - PyTorch is CPU-only")
        print("\nFor GPU support on RTX 5070 Ti:")
        print("  1. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
        print("  2. Install CUDA 12.1+: https://developer.nvidia.com/cuda-downloads")
        print("  3. Reinstall PyTorch:")
        print("     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print(f"\n[OK] CUDA detected: {torch.version.cuda}")
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("\n[INFO] PyTorch not installed yet")
    print("\nFor CPU-only (testing):")
    print("  pip install torch torchvision torchaudio")
    print("\nFor GPU support (RTX 5070 Ti):")
    print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

# Optional: Advanced ML
print("\n" + "="*70)
print("Optional: Advanced ML Libraries")
print("="*70)
print("\nThese are optional and can be installed later:")
print("  pip install transformers>=4.30.0")
print("  pip install stable-baselines3>=2.0.0")
print("  pip install gymnasium>=0.29.0")

# Summary
print("\n" + "="*70)
print("Installation Complete!")
print("="*70)
print("\nNext steps:")
print("  1. Setup API keys: python scripts/setup_api_keys.py --interactive")
print("  2. Test data pipeline: python scripts/test_data_pipeline.py")
print("  3. Verify GPU (if applicable): python scripts/verify_gpu_setup.py")
print("  4. Collect data: python scripts/collect_historical_data.py")
print("\nSee QUICK_START.md for detailed guide")
print("="*70)



