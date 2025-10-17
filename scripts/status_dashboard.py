#!/usr/bin/env python3
"""
Comprehensive Status Dashboard
Shows current status of all system components.
"""

import sys
import logging
from pathlib import Path
import json
import subprocess
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatusDashboard:
    """Comprehensive status dashboard for the AI trading system."""

    def __init__(self):
        self.status = {
            'timestamp': datetime.now().isoformat(),
            'gpu_status': {},
            'data_status': {},
            'model_status': {},
            'trading_status': {},
            'build_status': {}
        }

    def check_gpu_status(self):
        """Check GPU and CUDA status."""
        try:
            # Check nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(',')
                self.status['gpu_status'] = {
                    'available': True,
                    'name': gpu_info[0].strip(),
                    'driver_version': gpu_info[1].strip(),
                    'memory_total': int(gpu_info[2].strip()),
                    'utilization': int(gpu_info[3].strip()),
                    'temperature': int(gpu_info[4].strip())
                }
            else:
                self.status['gpu_status'] = {'available': False, 'error': 'nvidia-smi failed'}
                
        except Exception as e:
            self.status['gpu_status'] = {'available': False, 'error': str(e)}

    def check_cuda_status(self):
        """Check CUDA installation."""
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
                version = version_line.split('release')[1].strip().split(',')[0]
                self.status['gpu_status']['cuda_version'] = version
            else:
                self.status['gpu_status']['cuda_available'] = False
        except Exception as e:
            self.status['gpu_status']['cuda_available'] = False
            self.status['gpu_status']['cuda_error'] = str(e)

    def check_pytorch_status(self):
        """Check PyTorch installation and GPU support."""
        try:
            import torch
            self.status['gpu_status']['pytorch_available'] = True
            self.status['gpu_status']['pytorch_version'] = torch.__version__
            self.status['gpu_status']['cuda_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                self.status['gpu_status']['cuda_device_count'] = torch.cuda.device_count()
                self.status['gpu_status']['cuda_device_name'] = torch.cuda.get_device_name(0)
                self.status['gpu_status']['cuda_compute_capability'] = torch.cuda.get_device_capability(0)
        except ImportError:
            self.status['gpu_status']['pytorch_available'] = False
        except Exception as e:
            self.status['gpu_status']['pytorch_error'] = str(e)

    def check_data_status(self):
        """Check data collection status."""
        try:
            data_dir = Path("data/historical/real_aster_only")
            if data_dir.exists():
                files = list(data_dir.glob("*.parquet"))
                total_size = sum(f.stat().st_size for f in files) / (1024*1024)  # MB
                
                self.status['data_status'] = {
                    'available': True,
                    'file_count': len(files),
                    'total_size_mb': round(total_size, 2),
                    'files': [f.name for f in files[:10]]  # First 10 files
                }
            else:
                self.status['data_status'] = {'available': False, 'error': 'Data directory not found'}
                
        except Exception as e:
            self.status['data_status'] = {'available': False, 'error': str(e)}

    def check_model_status(self):
        """Check trained model status."""
        try:
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pth"))
                
                self.status['model_status'] = {
                    'available': len(model_files) > 0,
                    'model_count': len(model_files),
                    'models': [f.name for f in model_files]
                }
            else:
                self.status['model_status'] = {'available': False, 'error': 'Models directory not found'}
                
        except Exception as e:
            self.status['model_status'] = {'available': False, 'error': str(e)}

    def check_build_status(self):
        """Check PyTorch build status."""
        try:
            pytorch_dir = Path("D:/CodingFiles/pytorch")
            build_dir = pytorch_dir / "build"
            
            if build_dir.exists():
                build_size = sum(f.stat().st_size for f in build_dir.rglob('*') if f.is_file()) / (1024*1024*1024)  # GB
                self.status['build_status'] = {
                    'in_progress': True,
                    'build_size_gb': round(build_size, 2),
                    'build_directory': str(build_dir)
                }
            else:
                self.status['build_status'] = {
                    'in_progress': False,
                    'status': 'Build directory not found'
                }
                
        except Exception as e:
            self.status['build_status'] = {'error': str(e)}

    def check_trading_status(self):
        """Check trading bot status."""
        try:
            # Check if trading bot is running
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                  capture_output=True, text=True, timeout=10)
            
            trading_running = 'ai_trading_bot.py' in result.stdout
            
            self.status['trading_status'] = {
                'bot_running': trading_running,
                'paper_trading_available': Path("trading/ai_trading_bot.py").exists()
            }
            
        except Exception as e:
            self.status['trading_status'] = {'error': str(e)}

    def generate_report(self):
        """Generate comprehensive status report."""
        print("""
================================================================================
                        AI Trading System Status Dashboard
================================================================================
        """)
        
        # GPU Status
        print("🖥️  GPU Status:")
        gpu = self.status['gpu_status']
        if gpu.get('available'):
            print(f"   ✅ GPU: {gpu.get('name', 'Unknown')}")
            print(f"   ✅ Driver: {gpu.get('driver_version', 'Unknown')}")
            print(f"   ✅ Memory: {gpu.get('memory_total', 0)} MB")
            print(f"   ✅ Utilization: {gpu.get('utilization', 0)}%")
            print(f"   ✅ Temperature: {gpu.get('temperature', 0)}°C")
            print(f"   ✅ CUDA: {gpu.get('cuda_version', 'Unknown')}")
            print(f"   ✅ PyTorch: {gpu.get('pytorch_version', 'Not installed')}")
            print(f"   ✅ CUDA Available: {gpu.get('cuda_available', False)}")
        else:
            print(f"   ❌ GPU: {gpu.get('error', 'Not available')}")
        
        print()
        
        # Data Status
        print("📊 Data Status:")
        data = self.status['data_status']
        if data.get('available'):
            print(f"   ✅ Files: {data.get('file_count', 0)}")
            print(f"   ✅ Size: {data.get('total_size_mb', 0)} MB")
            print(f"   ✅ Directory: data/historical/real_aster_only")
        else:
            print(f"   ❌ Data: {data.get('error', 'Not available')}")
        
        print()
        
        # Model Status
        print("🤖 Model Status:")
        models = self.status['model_status']
        if models.get('available'):
            print(f"   ✅ Models: {models.get('model_count', 0)}")
            print(f"   ✅ Files: {', '.join(models.get('models', [])[:5])}")
        else:
            print(f"   ❌ Models: {models.get('error', 'Not available')}")
        
        print()
        
        # Build Status
        print("🔨 Build Status:")
        build = self.status['build_status']
        if build.get('in_progress'):
            print(f"   🔄 PyTorch build in progress")
            print(f"   📁 Size: {build.get('build_size_gb', 0)} GB")
        else:
            print(f"   ⏸️  PyTorch build: {build.get('status', 'Unknown')}")
        
        print()
        
        # Trading Status
        print("💰 Trading Status:")
        trading = self.status['trading_status']
        if trading.get('bot_running'):
            print(f"   ✅ Trading bot: Running")
        else:
            print(f"   ⏸️  Trading bot: Stopped")
        print(f"   ✅ Paper trading: Available")
        
        print()
        
        # Summary
        print("📋 Summary:")
        gpu_ok = gpu.get('available') and gpu.get('cuda_available')
        data_ok = data.get('available')
        models_ok = models.get('available')
        
        if gpu_ok and data_ok and models_ok:
            print("   🎉 System ready for AI trading!")
        elif gpu_ok and data_ok:
            print("   ⚠️  GPU and data ready, need to train models")
        elif data_ok:
            print("   ⚠️  Data ready, need GPU setup and model training")
        else:
            print("   ❌ System not ready - check individual components")

    def run_checks(self):
        """Run all status checks."""
        logger.info("Running status checks...")
        
        self.check_gpu_status()
        self.check_cuda_status()
        self.check_pytorch_status()
        self.check_data_status()
        self.check_model_status()
        self.check_build_status()
        self.check_trading_status()
        
        self.status['timestamp'] = datetime.now().isoformat()
        
        # Save status to file
        with open('system_status.json', 'w') as f:
            json.dump(self.status, f, indent=2)
        
        logger.info("Status checks completed")


def main():
    """Main execution."""
    dashboard = StatusDashboard()
    dashboard.run_checks()
    dashboard.generate_report()


if __name__ == "__main__":
    main()
