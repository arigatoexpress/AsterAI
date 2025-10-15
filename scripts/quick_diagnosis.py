"""
Quick diagnosis script to check current setup status
"""

import sys
import subprocess
from pathlib import Path

print("""
╔════════════════════════════════════════════════════════════════╗
║           AsterAI - Quick Diagnosis                           ║
╚════════════════════════════════════════════════════════════════╝
""")

def check_package(package_name, display_name=None):
    """Check if a package is installed"""
    if display_name is None:
        display_name = package_name

    try:
        __import__(package_name)
        return f"[OK] {display_name}"
    except ImportError:
        return f"[--] {display_name}"

def check_pytorch():
    """Special check for PyTorch CUDA status"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        version = torch.__version__

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            return f"[OK] PyTorch {version} (CUDA: {torch.version.cuda}, GPU: {gpu_name})"
        else:
            return f"[--] PyTorch {version} (CPU-only, no CUDA)"
    except ImportError:
        return "[--] PyTorch not installed"

print("\n" + "="*70)
print("API KEYS STATUS")
print("="*70)

try:
    api_keys_file = Path('.api_keys.json')
    if api_keys_file.exists():
        import json
        with open(api_keys_file, 'r') as f:
            keys = json.load(f)

        print("API Keys file found:")
        for key, value in keys.items():
            if key in ['aster_api_key', 'aster_secret_key', 'alpha_vantage_key', 'fred_api_key', 'finnhub_key', 'newsapi_key']:
                status = "[OK]" if value and value != "Not set" else "[--]"
            else:
                status = "[--]"  # Optional keys
            print(f"  {status} {key}: {'***' if value and value != 'Not set' else 'Not set'}")
    else:
        print("[--] No .api_keys.json file found")
except Exception as e:
    print(f"[ERROR] {e}")

print("\n" + "="*70)
print("PACKAGE STATUS")
print("="*70)

# Core packages
packages = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('requests', 'Requests'),
    ('aiohttp', 'AIOHTTP'),
    ('yfinance', 'Yahoo Finance'),
    ('tenacity', 'Tenacity'),
    ('pyarrow', 'PyArrow'),
    ('pandas_ta', 'Pandas TA'),
    ('ta', 'TA Library'),
    ('fastapi', 'FastAPI'),
    ('uvicorn', 'Uvicorn'),
    ('streamlit', 'Streamlit'),
    ('plotly', 'Plotly'),
    ('scikit_learn', 'Scikit-learn'),
]

for package, name in packages:
    print(check_package(package, name))

# Special PyTorch check
print(check_pytorch())

print("\n" + "="*70)
print("SYSTEM INFO")
print("="*70)

import platform
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print(f"Architecture: {platform.machine()}")

print("\n" + "="*70)
print("CURRENT STATUS")
print("="*70)

# Count OK packages
total_packages = len(packages) + 1  # +1 for PyTorch special check
ok_count = 0
for package, name in packages:
    if check_package(package, name).startswith("[OK]"):
        ok_count += 1

# Check API keys
api_ok_count = 0
try:
    if api_keys_file.exists():
        with open(api_keys_file, 'r') as f:
            keys = json.load(f)
        required_keys = ['aster_api_key', 'aster_secret_key', 'alpha_vantage_key', 'fred_api_key']
        for key in required_keys:
            if keys.get(key) and keys[key] != "Not set":
                api_ok_count += 1
except:
    pass

print(f"Core Packages: {ok_count}/{total_packages} installed")
print(f"API Keys: {api_ok_count}/4 required configured")
print(f"PyTorch CUDA: {'YES' if 'CUDA' in check_pytorch() else 'NO'}")

overall_status = "READY" if ok_count >= total_packages * 0.8 and api_ok_count >= 4 else "NEEDS FIXING"

print(f"\nOverall Status: {overall_status}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

if overall_status != "READY":
    print("1. Install missing packages:")
    print("   python scripts/install_dependencies.py")
    print("")
    print("2. Fix PyTorch CUDA:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("")
    print("3. Verify setup:")
    print("   python scripts/quick_diagnosis.py")

print("\n4. Test data pipeline:")
print("   python scripts/test_data_pipeline.py")

print("\n5. Collect historical data:")
print("   python scripts/collect_historical_data.py --start-date 2024-01-01 --priority 2")

print("="*70)



