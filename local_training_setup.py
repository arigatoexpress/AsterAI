#!/usr/bin/env python3
"""
RTX 5070Ti + 16-core AMD Ryzen Local ML Training Setup
Optimized for maximum performance in AI trading model development.
"""

import subprocess
import sys
import os
import multiprocessing
import platform
from pathlib import Path
import json
import time


class LocalTrainingEnvironment:
    """Set up optimized local training environment for RTX 5070Ti + 16-core AMD."""

    def __init__(self):
        self.hardware = self._detect_hardware()
        self.base_dir = Path.home() / "ai_trading_local"
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.notebooks_dir = self.base_dir / "notebooks"
        self.results_dir = self.base_dir / "results"

    def _detect_hardware(self):
        """Detect and validate hardware."""
        hardware = {
            'cpu_cores': multiprocessing.cpu_count(),
            'platform': platform.platform(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}"
        }

        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                hardware.update({
                    'gpu_available': True,
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    'cuda_version': torch.version.cuda
                })
        except:
            hardware['gpu_available'] = False

        return hardware

    def setup_directories(self):
        """Create optimized directory structure."""
        dirs = [
            self.models_dir, self.data_dir, self.notebooks_dir, self.results_dir,
            self.models_dir / "lstm", self.models_dir / "transformers", self.models_dir / "rl",
            self.data_dir / "raw", self.data_dir / "processed", self.data_dir / "features",
            self.results_dir / "backtests", self.results_dir / "benchmarks"
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"✅ Created training directories in {self.base_dir}")

    def install_optimized_environment(self):
        """Install RTX 5070Ti optimized environment."""
        print("🚀 Installing RTX 5070Ti + 16-core AMD Optimized Environment")

        # Install PyTorch with CUDA 12.1 (RTX 5070Ti compatible)
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch>=2.1.0+cu121",
                "torchvision>=0.16.0+cu121",
                "torchaudio>=2.1.0+cu121",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], check=True)
            print("✅ PyTorch with CUDA 12.1 installed")
        except subprocess.CalledProcessError:
            print("⚠️  PyTorch CUDA installation failed - using CPU version")
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])

        # Install GPU-accelerated libraries
        packages = [
            "xgboost", "lightgbm", "catboost",  # GPU ML
            "cudf-cu11", "cuml-cu11",  # RAPIDS GPU DataFrames
            "stable-baselines3", "gymnasium",  # RL
            "optuna", "hyperopt",  # Hyperparameter optimization
            "ta-lib", "pandas-ta",  # Technical analysis
            "yfinance", "ccxt",  # Market data
            "tweepy", "newsapi-python",  # Social/news data
            "jupyterlab", "notebook",  # Development
            "plotly", "matplotlib", "seaborn",  # Visualization
            "pytest", "pytest-benchmark",  # Testing
            "black", "isort", "mypy"  # Code quality
        ]

        for package in packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️  Failed to install {package}")

    def optimize_system_settings(self):
        """Apply hardware-specific optimizations."""
        print("⚙️  Applying Hardware Optimizations")

        # Set environment variables
        os.environ['OMP_NUM_THREADS'] = str(self.hardware['cpu_cores'])
        os.environ['MKL_NUM_THREADS'] = str(self.hardware['cpu_cores'])
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.hardware['cpu_cores'])
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.hardware['cpu_cores'])
        os.environ['TORCH_NUM_THREADS'] = str(self.hardware['cpu_cores'])

        if self.hardware['gpu_available']:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'  # RTX 5070Ti

        # Create optimization config
        config = {
            'hardware': self.hardware,
            'training': {
                'batch_size_per_gpu': 128,
                'num_workers': self.hardware['cpu_cores'],
                'mixed_precision': self.hardware['gpu_available'],
                'gradient_checkpointing': True,
                'compile_models': self.hardware['gpu_available']
            },
            'data_loading': {
                'num_workers': min(8, self.hardware['cpu_cores']),
                'pin_memory': self.hardware['gpu_available'],
                'prefetch_factor': 4
            }
        }

        config_file = self.base_dir / "hardware_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print("✅ Hardware optimizations applied")

    def create_jupyter_config(self):
        """Create optimized Jupyter configuration."""
        jupyter_config = {
            "NotebookApp": {
                "ip": "0.0.0.0",
                "port": 8888,
                "port_retries": 0,
                "open_browser": False,
                "allow_root": True,
                "token": "",
                "password": "",
                "allow_origin": "*",
                "notebook_dir": str(self.notebooks_dir),
                "max_buffer_size": 1073741824,  # 1GB
                "websocket_compression_options": {},
                "tornado_settings": {
                    "max_buffer_size": 1073741824
                }
            },
            "MultiKernelManager": {
                "default_kernel_name": "python3"
            }
        }

        config_dir = Path.home() / ".jupyter"
        config_dir.mkdir(exist_ok=True)

        import yaml
        with open(config_dir / "jupyter_notebook_config.py", 'w') as f:
            f.write(f"c = {repr(jupyter_config)}\n")

        print("✅ Jupyter configuration optimized for RTX 5070Ti")

    def run_benchmarks(self):
        """Run comprehensive hardware benchmarks."""
        print("🔬 Running Hardware Benchmarks")

        import torch
        import numpy as np
        import time

        benchmarks = {}

        # CPU benchmark
        print("Testing CPU performance...")
        size = 2000
        a = np.random.random((size, size))
        b = np.random.random((size, size))

        start = time.time()
        c = np.dot(a, b)
        cpu_time = time.time() - start

        benchmarks['cpu_matrix_mult'] = {
            'size': size,
            'time': cpu_time,
            'gflops': (2 * size**3) / (cpu_time * 1e9)
        }

        # GPU benchmark
        if self.hardware['gpu_available']:
            print("Testing GPU performance...")
            device = torch.device('cuda')

            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)

            # Warm up
            for _ in range(3):
                z = torch.matmul(x, y)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(5):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) / 5

            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            benchmarks['gpu_matrix_mult'] = {
                'size': size,
                'time': gpu_time,
                'speedup': speedup,
                'gflops': (2 * size**3) / (gpu_time * 1e9) if gpu_time > 0 else 0
            }

        # Save benchmarks
        benchmark_file = self.results_dir / "hardware_benchmarks.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmarks, f, indent=2)

        print("✅ Benchmarks complete")
        return benchmarks


def main():
    """Setup complete local training environment."""
    env = LocalTrainingEnvironment()

    print("🚀 Setting Up RTX 5070Ti + 16-core AMD Training Environment")
    print("=" * 60)

    # Setup directories
    env.setup_directories()

    # Install optimized environment
    env.install_optimized_environment()

    # Apply optimizations
    env.optimize_system_settings()

    # Configure Jupyter
    env.create_jupyter_config()

    # Run benchmarks
    benchmarks = env.run_benchmarks()

    print("\n" + "=" * 60)
    print("🎉 Local Training Environment Setup Complete!")
    print("=" * 60)

    print(f"Hardware: {env.hardware['cpu_cores']} CPU cores")
    if env.hardware['gpu_available']:
        print(f"GPU: {env.hardware['gpu_name']} ({env.hardware['gpu_memory']:.1f}GB)")
        if 'gpu_matrix_mult' in benchmarks:
            speedup = benchmarks['gpu_matrix_mult']['speedup']
            print(f"Performance: {speedup:.1f}x GPU acceleration")

    print("
📁 Directories created:"
    print(f"   Models: {env.models_dir}")
    print(f"   Data: {env.data_dir}")
    print(f"   Notebooks: {env.notebooks_dir}")
    print(f"   Results: {env.results_dir}")

    print("
🚀 Ready for high-performance AI training!"
    print("   Next: cd ~/ai_trading_local/notebooks")
    print("   Run: jupyter lab --no-browser")

    return 0


if __name__ == "__main__":
    exit(main())
