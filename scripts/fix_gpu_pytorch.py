#!/usr/bin/env python3
"""
Fix PyTorch for RTX 5070 Ti (Blackwell Architecture)
Installs PyTorch nightly build with CUDA 12.4+ support
"""

import subprocess
import sys
import platform

def check_current_pytorch():
    """Check current PyTorch installation"""
    try:
        import torch
        print(f"Current PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            try:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            except:
                print("GPU detection failed")
        return True
    except ImportError:
        print("PyTorch not installed")
        return False

def uninstall_pytorch():
    """Uninstall current PyTorch"""
    print("\n" + "="*60)
    print("STEP 1: Uninstalling current PyTorch...")
    print("="*60)
    
    packages = ['torch', 'torchvision', 'torchaudio']
    for package in packages:
        print(f"Uninstalling {package}...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', package])

def install_pytorch_nightly():
    """Install PyTorch nightly with CUDA 12.4 support"""
    print("\n" + "="*60)
    print("STEP 2: Installing PyTorch Nightly (CUDA 12.4)")
    print("="*60)
    print("\nThis will install PyTorch with support for RTX 5070 Ti")
    print("Download size: ~2-3 GB")
    print("This may take 5-10 minutes...\n")
    
    # PyTorch nightly with CUDA 12.4
    cmd = [
        sys.executable, '-m', 'pip', 'install', '--pre',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/nightly/cu124'
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    return result.returncode == 0

def verify_installation():
    """Verify PyTorch can use the GPU"""
    print("\n" + "="*60)
    print("STEP 3: Verifying GPU Support")
    print("="*60)
    
    try:
        import torch
        print(f"\n✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test tensor on GPU
            print("\nTesting GPU tensor operations...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✅ GPU tensor operations working!")
            
            return True
        else:
            print("❌ CUDA not available")
            print("\nPossible issues:")
            print("1. NVIDIA drivers not installed")
            print("2. CUDA toolkit not installed")
            print("3. GPU not detected")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║          Fix PyTorch for RTX 5070 Ti                           ║
║          Install PyTorch Nightly with CUDA 12.4                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check current installation
    print("\nChecking current PyTorch installation...")
    check_current_pytorch()
    
    # Ask for confirmation
    print("\n" + "="*60)
    response = input("\nProceed with PyTorch reinstallation? (yes/no): ").lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Uninstall current PyTorch
    uninstall_pytorch()
    
    # Install PyTorch nightly
    success = install_pytorch_nightly()
    
    if not success:
        print("\n❌ Installation failed!")
        print("\nTry manual installation:")
        print("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
        return
    
    # Verify installation
    if verify_installation():
        print("\n" + "="*60)
        print("✅ SUCCESS! PyTorch is now configured for RTX 5070 Ti")
        print("="*60)
        print("\nYou can now run GPU-accelerated training:")
        print("  python scripts/quick_train_model.py")
    else:
        print("\n" + "="*60)
        print("⚠️  Installation completed but GPU not detected")
        print("="*60)
        print("\nPlease check:")
        print("1. NVIDIA drivers are up to date")
        print("2. CUDA 12.4+ is installed")
        print("3. GPU is properly connected")

if __name__ == "__main__":
    main()


