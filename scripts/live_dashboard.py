#!/usr/bin/env python3
"""
Live Dashboard for GPU Setup and Data Quality Progress
Automatically refreshes and shows real-time status
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_process_info():
    """Get information about running processes"""
    processes = {}

    try:
        # Check for Python processes
        python_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            if 'python' in proc.info['name'].lower():
                try:
                    proc.info['cpu'] = proc.cpu_percent(interval=0.1)
                    proc.info['memory_mb'] = proc.memory_info().rss / 1024 / 1024
                    python_procs.append(proc.info)
                except:
                    pass
        processes['python'] = python_procs

        # Check for CMake processes
        cmake_procs = []
        for proc in psutil.process_iter(['pid', 'name']):
            if 'cmake' in proc.info['name'].lower():
                cmake_procs.append(proc.info)
        processes['cmake'] = cmake_procs

        # Check for CL compiler
        cl_procs = []
        for proc in psutil.process_iter(['pid', 'name']):
            if 'cl.exe' in proc.info['name'].lower():
                cl_procs.append(proc.info)
        processes['cl'] = cl_procs

    except Exception as e:
        processes['error'] = str(e)

    return processes

def get_gpu_info():
    """Get GPU information"""
    gpu_info = {}

    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = [part.strip() for part in lines[0].split(',')]
                if len(parts) >= 6:
                    gpu_info = {
                        'name': parts[0],
                        'driver': parts[1],
                        'memory_total': f"{parts[2]} MB",
                        'memory_used': f"{parts[3]} MB",
                        'temperature': f"{parts[4]}°C",
                        'utilization': f"{parts[5]}%"
                    }
    except:
        gpu_info = {'error': 'NVIDIA GPU not detected'}

    return gpu_info

def get_cuda_info():
    """Get CUDA information"""
    cuda_info = {}

    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_info['version'] = line.split('release')[1].split(',')[0].strip()
                    break
        else:
            cuda_info['version'] = 'Not found'
    except:
        cuda_info['version'] = 'Not available'

    return cuda_info

def get_data_quality_status():
    """Get data quality improvement status"""
    status = {
        'running': False,
        'progress': 'Unknown',
        'issues_found': 0,
        'fixes_applied': 0,
        'last_update': 'Never'
    }

    # Check if data quality script is running
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'improve_data_quality.py' in cmdline:
                    status['running'] = True
                    break
    except:
        pass

    # Check for quality report
    report_file = Path('data/historical/ultimate_dataset/crypto/data_quality_report.json')
    if report_file.exists():
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
                status['issues_found'] = len(data.get('issues_found', []))
                status['fixes_applied'] = len(data.get('fixes_applied', []))
                status['last_update'] = data.get('timestamp', 'Unknown')
                status['progress'] = f"{len(data.get('fixes_applied', []))}/{len(data.get('issues_found', []))} issues fixed"
        except:
            status['progress'] = 'Error reading report'

    return status

def get_pytorch_build_status():
    """Get PyTorch build status"""
    status = {
        'running': False,
        'stage': 'Not started',
        'progress': '0%',
        'last_log': 'No logs yet'
    }

    # Check if build is running
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'setup.py develop' in cmdline or 'build_pytorch' in cmdline:
                    status['running'] = True
                    break
    except:
        pass

    # Check build logs
    log_dir = Path('logs')
    if log_dir.exists():
        log_files = list(log_dir.glob('pytorch_build_*.log'))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        status['last_log'] = lines[-1].strip()[:80] + '...' if len(lines[-1]) > 80 else lines[-1].strip()

                        # Try to determine stage
                        for line in reversed(lines[-20:]):
                            if 'cmake' in line.lower():
                                status['stage'] = 'CMake Configuration'
                                break
                            elif 'building' in line.lower():
                                status['stage'] = 'Compiling'
                                break
                            elif 'successfully installed' in line.lower():
                                status['stage'] = 'Complete'
                                status['progress'] = '100%'
                                break
            except:
                status['last_log'] = 'Error reading log'

    return status

def display_dashboard():
    """Display the live dashboard"""
    while True:
        clear_screen()

        print("="*80)
        print("🎯 LIVE DASHBOARD - GPU Setup & Data Quality Progress")
        print("="*80)
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # GPU Information
        print("🖥️  GPU STATUS")
        print("-"*40)
        gpu_info = get_gpu_info()
        if 'error' not in gpu_info:
            print(f"GPU Name: {gpu_info['name']}")
            print(f"Driver: {gpu_info['driver']}")
            print(f"Memory: {gpu_info['memory_used']} / {gpu_info['memory_total']}")
            print(f"Temperature: {gpu_info['temperature']}")
            print(f"Utilization: {gpu_info['utilization']}")
        else:
            print(f"❌ {gpu_info['error']}")

        # CUDA Information
        cuda_info = get_cuda_info()
        print(f"CUDA Version: {cuda_info['version']}")
        print()

        # Process Information
        print("⚙️  ACTIVE PROCESSES")
        print("-"*40)
        processes = get_process_info()

        if 'error' not in processes:
            proc_counts = {
                'Python': len(processes.get('python', [])),
                'CMake': len(processes.get('cmake', [])),
                'CL Compiler': len(processes.get('cl', []))
            }

            for proc_type, count in proc_counts.items():
                status = f"✅ {count} running" if count > 0 else "⏸️  None"
                print(f"{proc_type}: {status}")
        else:
            print(f"❌ Error getting process info: {processes['error']}")

        print()

        # Data Quality Status
        print("📊 DATA QUALITY IMPROVEMENT")
        print("-"*40)
        dq_status = get_data_quality_status()
        print(f"Status: {'🟢 Running' if dq_status['running'] else '⏸️  Not running'}")
        print(f"Progress: {dq_status['progress']}")
        print(f"Issues Found: {dq_status['issues_found']}")
        print(f"Fixes Applied: {dq_status['fixes_applied']}")
        print(f"Last Update: {dq_status['last_update']}")
        print()

        # PyTorch Build Status
        print("🔧 PYTORCH BUILD STATUS")
        print("-"*40)
        build_status = get_pytorch_build_status()
        print(f"Status: {'🟢 Running' if build_status['running'] else '⏸️  Not running'}")
        print(f"Stage: {build_status['stage']}")
        print(f"Progress: {build_status['progress']}")
        print(f"Last Log: {build_status['last_log']}")
        print()

        # Overall Progress
        print("📈 OVERALL PROGRESS")
        print("-"*40)

        # Calculate overall progress
        progress_items = [
            ("CUDA 12.8", cuda_info['version'] != 'Not available'),
            ("Conda Environment", True),  # Assume exists
            ("PyTorch Repository", Path("D:\CodingFiles\pytorch").exists()),
            ("Data Quality", dq_status['fixes_applied'] > 0),
            ("PyTorch Build", build_status['stage'] == 'Complete')
        ]

        completed = sum(1 for _, done in progress_items if done)
        total = len(progress_items)
        overall_progress = completed / total * 100

        print(f"Overall Progress: {completed}/{total} steps ({overall_progress:.1f}%)")
        print()

        for item, done in progress_items:
            status = "✅" if done else "⏸️"
            print(f"  {status} {item}")

        print()
        print("🔄 Refreshing in 10 seconds... (Ctrl+C to exit)")
        print("="*80)

        time.sleep(10)

def main():
    """Main dashboard function"""
    try:
        display_dashboard()
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped by user")
        print("To restart: python scripts/live_dashboard.py")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")
        print("Try running individual status checks:")
        print("  python -c \"import torch; print(torch.cuda.is_available())\"")
        print("  python scripts/verify_gpu_build.py")

if __name__ == "__main__":
    main()

