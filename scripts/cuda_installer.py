#!/usr/bin/env python3
"""
Comprehensive CUDA and PyTorch Installation Script for AsterAI
Handles CUDA toolkit installation and PyTorch setup with multiple fallback options.
"""

import subprocess
import sys
import os
import urllib.request
import json
import platform
from pathlib import Path


class CUDAInstaller:
    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        print("=== CUDA & PyTorch Installation Diagnostics ===")
        print(f"System: {self.system}")
        print(f"Architecture: {self.machine}")
        print(f"Python version: {self.python_version}")
        print()

    def run_command(self, cmd, description=""):
        """Run a command and return success status"""
        try:
            print(f"🔄 {description}")
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                return True
            else:
                print(f"❌ {description} - FAILED")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ {description} - EXCEPTION: {e}")
            return False

    def check_nvidia_drivers(self):
        """Check if NVIDIA drivers are installed"""
        print("🔍 Checking NVIDIA drivers...")
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Driver Version' in line:
                        driver_version = line.split('Driver Version:')[1].split()[0]
                        print(f"✅ NVIDIA drivers found: {driver_version}")
                    if 'CUDA Version' in line:
                        cuda_version = line.split('CUDA Version:')[1].split()[0]
                        print(f"✅ CUDA version (drivers): {cuda_version}")
                        return cuda_version
            else:
                print("❌ nvidia-smi failed")
                return None
        except Exception as e:
            print(f"❌ NVIDIA driver check failed: {e}")
            return None

    def install_cuda_toolkit(self):
        """Install CUDA toolkit"""
        cuda_version = self.check_nvidia_drivers()
        if not cuda_version:
            print("❌ No NVIDIA drivers found. Please install NVIDIA drivers first.")
            return False

        print(f"\n🔧 Installing CUDA toolkit for driver version {cuda_version}")

        # For CUDA 13.0 drivers, we need CUDA 12.4 toolkit (most compatible)
        toolkit_version = "12.4"

        if self.system == "windows":
            # Download and install CUDA toolkit
            cuda_url = f"https://developer.download.nvidia.com/compute/cuda/{toolkit_version}/local_installers/cuda_{toolkit_version}.0_windows.exe"
            installer_path = f"C:\\Temp\\cuda_{toolkit_version}_installer.exe"

            os.makedirs("C:\\Temp", exist_ok=True)

            print(f"📥 Downloading CUDA {toolkit_version} toolkit...")
            try:
                urllib.request.urlretrieve(cuda_url, installer_path)
                print("✅ Download complete")

                # Run installer (this will require user interaction)
                print("🚀 Running CUDA installer...")
                print("⚠️  Please follow the installation wizard and select 'Custom Installation'")
                print("⚠️  Make sure to install CUDA Toolkit and CUDA Visual Studio Integration")
                print("⚠️  This may take several minutes...")

                result = subprocess.run([installer_path], shell=True)
                return result.returncode == 0

            except Exception as e:
                print(f"❌ CUDA toolkit installation failed: {e}")
                return False
        else:
            print("❌ CUDA toolkit installation not supported on this platform")
            return False

    def install_pytorch(self, cuda_version=None):
        """Install PyTorch with appropriate CUDA version"""
        print("\n🔧 Installing PyTorch...")

        # Check if we should use CUDA or CPU
        if cuda_version:
            print(f"🎯 Installing PyTorch with CUDA {cuda_version} support")
            if cuda_version.startswith("13."):
                # CUDA 13.0 - try nightly builds or fallback to 12.4
                print("⚠️  CUDA 13.0 detected. PyTorch may not have official support yet.")
                print("🔄 Trying CUDA 12.4 compatibility...")

                index_url = "https://download.pytorch.org/whl/cu124"
                torch_cmd = f"pip install torch torchvision torchaudio --index-url {index_url}"

            elif cuda_version.startswith("12."):
                index_url = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
                torch_cmd = f"pip install torch torchvision torchaudio --index-url {index_url}"
            else:
                print(f"⚠️  Unknown CUDA version {cuda_version}, falling back to CPU")
                cuda_version = None

        if not cuda_version:
            print("🎯 Installing PyTorch CPU version")
            torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

        success = self.run_command(torch_cmd, "Installing PyTorch")
        if success:
            self.test_pytorch_installation(cuda_version)
        return success

    def test_pytorch_installation(self, expected_cuda=None):
        """Test PyTorch installation"""
        print("\n🧪 Testing PyTorch installation...")
        test_code = '''
import torch
import sys
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version.split()[0]}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("Running on CPU")
'''
        try:
            result = subprocess.run([sys.executable, '-c', test_code],
                                  capture_output=True, text=True)
            print("PyTorch test results:")
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:")
                print(result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ PyTorch test failed: {e}")
            return False

    def install_conda_pytorch(self):
        """Alternative: Install PyTorch via conda"""
        print("\n🔧 Trying conda installation...")
        try:
            # Check if conda is available
            result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ conda not found")
                return False

            print("✅ conda found, installing PyTorch...")

            # Install PyTorch via conda
            conda_cmd = "conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y"
            success = self.run_command(conda_cmd, "Installing PyTorch via conda")

            if success:
                self.test_pytorch_installation("12.4")
            return success

        except Exception as e:
            print(f"❌ conda installation failed: {e}")
            return False

    def create_requirements_with_fallbacks(self):
        """Create requirements files with GPU and CPU fallbacks"""
        print("\n📝 Creating requirements files...")

        gpu_requirements = """
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
ipykernel>=6.0.0
gpustat>=1.0.0
nvitop>=1.0.0
"""

        cpu_requirements = """
torch>=2.0.0+cpu
torchvision>=0.15.0+cpu
torchaudio>=2.0.0+cpu
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
ipykernel>=6.0.0
"""

        with open("requirements-gpu.txt", "w") as f:
            f.write(gpu_requirements.strip())

        with open("requirements-cpu.txt", "w") as f:
            f.write(cpu_requirements.strip())

        print("✅ Created requirements-gpu.txt and requirements-cpu.txt")

    def run_installation(self):
        """Main installation workflow"""
        print("🚀 Starting CUDA & PyTorch Installation\n")

        # Step 1: Check current status
        cuda_version = self.check_nvidia_drivers()

        # Step 2: Try PyTorch installation with CUDA support
        if cuda_version:
            print(f"\n🎯 CUDA {cuda_version} detected, attempting PyTorch with CUDA support...")

            # First try direct pip installation
            if not self.install_pytorch(cuda_version):
                print("\n🔄 PyTorch with CUDA failed, trying conda installation...")
                if not self.install_conda_pytorch():
                    print("\n🔄 All CUDA installations failed, falling back to CPU...")
                    self.install_pytorch()  # CPU fallback
        else:
            print("\n⚠️  No CUDA detected, installing CPU version...")
            self.install_pytorch()

        # Step 3: Create requirements files
        self.create_requirements_with_fallbacks()

        # Step 4: Final verification
        print("\n🎯 Installation complete!")
        print("📋 Next steps:")
        print("1. Restart your Python session/IDE")
        print("2. Run: python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"")
        print("3. If GPU training doesn't work, use: pip install -r requirements-cpu.txt")

        return True


def main():
    installer = CUDAInstaller()
    installer.run_installation()


if __name__ == "__main__":
    main()

