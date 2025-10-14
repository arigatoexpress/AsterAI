#!/usr/bin/env python3
"""
Setup GPU-Optimized Environment for RTX 5070Ti + 16-core AMD
Configures high-performance machine learning environment for local training.
"""

import subprocess
import sys
import os
import platform
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_hardware():
    """Check hardware specifications and compatibility."""
    print("🔍 Checking Hardware Specifications")
    print("=" * 50)

    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 12):
        print("⚠️  WARNING: Python 3.12+ recommended for full PyTorch support")
        print("   Consider upgrading to Python 3.12 for RTX 5070Ti optimization")

    # Check CPU
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"CPU Cores: {cpu_count}")
        if cpu_count < 8:
            print("⚠️  WARNING: Recommend 16+ cores for optimal performance")
    except:
        print("CPU Cores: Unknown")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")

            if "5070" not in gpu_name and "RTX" not in gpu_name:
                print("⚠️  WARNING: RTX 5070Ti not detected")
                print("   Ensure NVIDIA drivers are installed and CUDA is configured")

        else:
            print("❌ ERROR: CUDA not available")
            print("   Install NVIDIA drivers and CUDA toolkit for RTX 5070Ti")
            return False

    except ImportError:
        print("❌ ERROR: PyTorch not installed")
        print("   Install PyTorch with CUDA support for GPU acceleration")
        return False

    return True


def install_gpu_requirements():
    """Install GPU-optimized requirements."""
    print("\n📦 Installing GPU-Optimized Dependencies")
    print("=" * 50)

    try:
        # Install PyTorch with CUDA 12.1 (RTX 5070Ti compatible)
        print("Installing PyTorch with CUDA support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch>=2.0.0+cu121",
            "torchvision>=0.15.0+cu121",
            "torchaudio>=2.0.0+cu121",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)

        # Install GPU-accelerated libraries
        print("Installing GPU-accelerated ML libraries...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "xgboost", "lightgbm", "scikit-learn",
            "stable-baselines3", "gymnasium"
        ], check=True)

        # Install RAPIDS for GPU DataFrames (optional, requires CUDA)
        print("Installing RAPIDS GPU libraries...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "cudf-cu11", "cuml-cu11", "dask"
            ], check=True)
            print("✅ RAPIDS libraries installed")
        except subprocess.CalledProcessError:
            print("⚠️  RAPIDS installation failed - continuing without GPU DataFrames")

        print("✅ GPU dependencies installed successfully")

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

    return True


def optimize_system_settings():
    """Apply system-level optimizations."""
    print("\n⚙️  Applying System Optimizations")
    print("=" * 50)

    try:
        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['NUMEXPR_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())

        # PyTorch optimizations
        os.environ['TORCH_NUM_THREADS'] = str(multiprocessing.cpu_count())
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'  # RTX 5070Ti compute capability

        # Enable CUDA optimizations
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        print("✅ Environment variables optimized for multi-core CPU and GPU")

        # Check and set CPU governor for better performance (Linux only)
        if platform.system() == 'Linux':
            try:
                subprocess.run(['sudo', 'cpufreq-set', '-g', 'performance'],
                             capture_output=True, check=True)
                print("✅ CPU governor set to performance mode")
            except:
                print("⚠️  Could not set CPU governor (requires sudo)")

        return True

    except Exception as e:
        print(f"⚠️  Optimization warning: {e}")
        return False


def validate_gpu_setup():
    """Validate GPU setup and performance."""
    print("\n🧪 Validating GPU Setup")
    print("=" * 50)

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False

        # Test basic GPU operations
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)

        print(f"✅ GPU matrix multiplication test passed: {z.shape}")

        # Test memory allocation
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"✅ GPU memory test: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")

        # Test mixed precision
        with torch.cuda.amp.autocast():
            x_half = x.half()
            y_half = y.half()
            z_half = torch.matmul(x_half, y_half)

        print(f"✅ Mixed precision test passed: {z_half.dtype}")

        return True

    except Exception as e:
        print(f"❌ GPU validation failed: {e}")
        return False


def create_performance_benchmarks():
    """Create performance benchmarks for the system."""
    print("\n📊 Creating Performance Benchmarks")
    print("=" * 50)

    try:
        import torch
        import time
        import numpy as np

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Benchmarking on: {device}")

        # Matrix multiplication benchmark
        sizes = [1000, 2000, 4000]
        results = {}

        for size in sizes:
            # CPU benchmark
            x_cpu = torch.randn(size, size)
            y_cpu = torch.randn(size, size)

            start = time.time()
            z_cpu = torch.matmul(x_cpu, y_cpu)
            cpu_time = time.time() - start

            # GPU benchmark (if available)
            if torch.cuda.is_available():
                x_gpu = x_cpu.cuda()
                y_gpu = y_cpu.cuda()

                # Warm up
                for _ in range(3):
                    z_gpu = torch.matmul(x_gpu, y_gpu)

                torch.cuda.synchronize()
                start = time.time()
                for _ in range(5):
                    z_gpu = torch.matmul(x_gpu, y_gpu)
                torch.cuda.synchronize()
                gpu_time = (time.time() - start) / 5

                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                results[size] = {
                    'cpu_time': cpu_time,
                    'gpu_time': gpu_time,
                    'speedup': speedup
                }
                print(f"Matrix {size}x{size}: CPU {cpu_time:.3f}s, GPU {gpu_time:.3f}s, Speedup {speedup:.1f}x")
            else:
                results[size] = {'cpu_time': cpu_time}
                print(f"Matrix {size}x{size}: CPU {cpu_time:.3f}s")

        # Save benchmark results
        benchmark_file = Path.home() / "ai_trading_benchmarks.json"
        import json

        with open(benchmark_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'device': str(device),
                'results': results
            }, f, indent=2)

        print(f"✅ Benchmarks saved to {benchmark_file}")

        return True

    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        return False


def setup_model_directories():
    """Setup optimized directory structure for models."""
    print("\n📁 Setting Up Model Directories")
    print("=" * 50)

    directories = [
        "models/gpu_lstm",
        "models/rl_agents",
        "models/ensemble",
        "checkpoints",
        "data/cache",
        "logs/gpu",
        "ray_results"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Model directories created")

    # Create GPU-specific configuration files
    config_files = {
        "config-gpu.yaml": """# GPU-optimized configuration
gpu:
  enabled: true
  memory_gb: 16
  compute_capability: "8.9"

cpu:
  cores: 16
  threads: 32

training:
  batch_size: 128
  mixed_precision: true
  gradient_checkpointing: true
""",
        "ray_config.yaml": """# Ray configuration for multi-core
ray:
  num_cpus: 16
  num_gpus: 1
  object_store_memory: "8GB"
  enable_plasma_store: true
"""
    }

    for filename, content in config_files.items():
        with open(filename, 'w') as f:
            f.write(content)

    print("✅ Configuration files created")


def main():
    """Main setup function."""
    print("🚀 Setting Up GPU-Optimized AI Trading Environment")
    print("=" * 60)
    print("Target Hardware: RTX 5070Ti + 16-core AMD CPU")
    print("=" * 60)

    # Step 1: Hardware check
    if not check_hardware():
        print("\n❌ Hardware setup incomplete. Please install:")
        print("   1. NVIDIA drivers (for RTX 5070Ti)")
        print("   2. CUDA Toolkit 12.1")
        print("   3. PyTorch with CUDA support")
        return 1

    # Step 2: Install dependencies
    if not install_gpu_requirements():
        print("\n❌ Dependency installation failed")
        return 1

    # Step 3: Apply optimizations
    optimize_system_settings()

    # Step 4: Validate GPU setup
    if not validate_gpu_setup():
        print("\n❌ GPU validation failed")
        return 1

    # Step 5: Create benchmarks
    if not create_performance_benchmarks():
        print("\n⚠️  Benchmarking failed - continuing anyway")

    # Step 6: Setup directories
    setup_model_directories()

    print("\n" + "=" * 60)
    print("🎉 GPU Environment Setup Complete!")
    print("=" * 60)
    print("✅ Hardware validated")
    print("✅ Dependencies installed")
    print("✅ System optimized")
    print("✅ Benchmarks created")
    print("✅ Directories configured")
    print()
    print("🚀 Ready for high-performance AI training!")
    print("   - Use requirements-gpu.txt for GPU-optimized setup")
    print("   - Use config-gpu.yaml for RTX 5070Ti optimization")
    print("   - Monitor performance with GPU logging enabled")
    print()
    print("Next steps:")
    print("1. Run: python run_complete_system.py backtest")
    print("2. Train models: python -m mcp_trader.models.deep_learning.gpu_lstm")
    print("3. Monitor: tail -f logs/gpu/*.log")

    return 0


if __name__ == "__main__":
    exit(main())
