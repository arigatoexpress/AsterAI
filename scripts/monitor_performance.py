#!/usr/bin/env python3
"""
Real-time Performance Monitor for RTX 5070Ti + 16-core AMD
Monitors GPU utilization, memory usage, and training performance.
"""

import time
import psutil
import threading
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Real-time performance monitoring for AI training."""

    def __init__(self, log_dir: str = "logs/performance", interval: float = 1.0):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.running = False
        self.monitor_thread = None

        # Performance data
        self.metrics = []
        self.start_time = time.time()

    def start_monitoring(self):
        """Start real-time monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Performance monitoring started (interval: {self.interval}s)")

    def stop_monitoring(self):
        """Stop monitoring and save results."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self._save_metrics()
        logger.info("Performance monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)

                # Log to console every 10 seconds
                if len(self.metrics) % 10 == 0:
                    self._log_summary()

                time.sleep(self.interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        timestamp = time.time() - self.start_time

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()

        # Disk metrics
        disk = psutil.disk_usage('/')

        # Network metrics (optional)
        network = psutil.net_io_counters()

        # GPU metrics (if available)
        gpu_metrics = self._get_gpu_metrics()

        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()

        # Training-specific metrics (if available)
        training_metrics = self._get_training_metrics()

        metrics = {
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'cores': cpu_count,
                'load_per_core': [p for p in psutil.cpu_percent(interval=None, percpu=True)]
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'used_gb': disk.used / (1024**3),
                'percent': disk.percent
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'process': {
                'memory_mb': process_memory.rss / (1024**2),
                'cpu_percent': process_cpu,
                'threads': process.num_threads(),
                'open_files': len(process.open_files())
            },
            'training': training_metrics
        }

        # Merge GPU metrics
        if gpu_metrics:
            metrics['gpu'] = gpu_metrics

        return metrics

    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get NVIDIA GPU metrics."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None

            gpu_count = torch.cuda.device_count()
            gpu_metrics = {}

            for i in range(gpu_count):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_percent = (memory_allocated / (torch.cuda.get_device_properties(i).total_memory / 1024**3)) * 100

                gpu_metrics[f'gpu_{i}'] = {
                    'name': torch.cuda.get_device_name(i),
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                    'memory_percent': memory_percent,
                    'utilization_percent': self._get_gpu_utilization(i)
                }

            return gpu_metrics

        except Exception as e:
            return None

    def _get_gpu_utilization(self, device_id: int) -> float:
        """Get GPU utilization percentage."""
        try:
            # Try to use nvidia-ml-py3
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(device_id)
            util = nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            # Fallback to torch metrics if available
            try:
                import torch
                if torch.cuda.is_available():
                    # This is a rough estimate
                    return 50.0  # Placeholder
            except:
                pass
            return 0.0

    def _get_training_metrics(self) -> Dict[str, Any]:
        """Get training-specific metrics."""
        # This would be populated by training loops
        # For now, return empty dict or check for active training processes
        return {
            'active_training': False,
            'batch_size': None,
            'learning_rate': None,
            'loss': None,
            'epoch': None
        }

    def _log_summary(self):
        """Log performance summary."""
        if not self.metrics:
            return

        recent = self.metrics[-1]

        # CPU summary
        cpu_avg = np.mean(recent['cpu']['load_per_core'])

        # Memory summary
        memory_gb = recent['memory']['used_gb']

        # GPU summary (if available)
        gpu_summary = ""
        if 'gpu' in recent and recent['gpu']:
            gpu_info = list(recent['gpu'].values())[0]
            gpu_summary = f" | GPU: {gpu_info['memory_percent']:.1f}% ({gpu_info['memory_allocated_gb']:.1f}GB)"

        logger.info(f"Performance: CPU {cpu_avg:.1f}% avg, Mem {memory_gb:.1f}GB{gpu_summary}")

    def _save_metrics(self):
        """Save collected metrics to file."""
        if not self.metrics:
            return

        filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename

        # Summarize metrics for storage efficiency
        summary = {
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration': time.time() - self.start_time,
            'total_samples': len(self.metrics),
            'interval_seconds': self.interval,
            'summary': self._create_summary()
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Performance metrics saved to {filepath}")

    def _create_summary(self) -> Dict[str, Any]:
        """Create performance summary."""
        if not self.metrics:
            return {}

        # Calculate averages
        cpu_percents = [m['cpu']['percent'] for m in self.metrics]
        memory_percents = [m['memory']['percent'] for m in self.metrics]
        memory_used_gb = [m['memory']['used_gb'] for m in self.metrics]

        summary = {
            'cpu': {
                'avg_percent': np.mean(cpu_percents),
                'max_percent': np.max(cpu_percents),
                'min_percent': np.min(cpu_percents)
            },
            'memory': {
                'avg_percent': np.mean(memory_percents),
                'max_percent': np.max(memory_percents),
                'avg_used_gb': np.mean(memory_used_gb),
                'max_used_gb': np.max(memory_used_gb)
            },
            'samples': len(self.metrics),
            'duration_minutes': (time.time() - self.start_time) / 60
        }

        # GPU summary if available
        gpu_samples = []
        for metric in self.metrics:
            if 'gpu' in metric and metric['gpu']:
                gpu_info = list(metric['gpu'].values())[0]
                gpu_samples.append(gpu_info['memory_percent'])

        if gpu_samples:
            summary['gpu'] = {
                'avg_memory_percent': np.mean(gpu_samples),
                'max_memory_percent': np.max(gpu_samples)
            }

        return summary

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the most recent metrics."""
        return self.metrics[-1] if self.metrics else None

    def export_csv(self, filepath: str):
        """Export metrics to CSV format."""
        if not self.metrics:
            return

        # Flatten nested structure for CSV
        flat_metrics = []

        for metric in self.metrics:
            flat = {
                'timestamp': metric['timestamp'],
                'datetime': metric['datetime'],
                'cpu_percent': metric['cpu']['percent'],
                'cpu_frequency': metric['cpu']['frequency_mhz'],
                'memory_percent': metric['memory']['percent'],
                'memory_used_gb': metric['memory']['used_gb'],
                'disk_percent': metric['disk']['percent'],
                'process_memory_mb': metric['process']['memory_mb'],
                'process_cpu_percent': metric['process']['cpu_percent']
            }

            # Add GPU metrics if available
            if 'gpu' in metric and metric['gpu']:
                gpu_info = list(metric['gpu'].values())[0]
                flat.update({
                    'gpu_memory_percent': gpu_info['memory_percent'],
                    'gpu_memory_allocated_gb': gpu_info['memory_allocated_gb']
                })

            flat_metrics.append(flat)

        df = pd.DataFrame(flat_metrics)
        df.to_csv(filepath, index=False)
        logger.info(f"Metrics exported to {filepath}")


def monitor_training_session(duration_minutes: int = 60):
    """Monitor a training session for specified duration."""
    monitor = PerformanceMonitor(interval=2.0)  # More frequent for training

    print(f"📊 Starting performance monitoring for {duration_minutes} minutes")
    print("Press Ctrl+C to stop early")

    try:
        monitor.start_monitoring()

        # Monitor for specified duration
        end_time = time.time() + (duration_minutes * 60)
        while time.time() < end_time:
            time.sleep(10)  # Update every 10 seconds

            # Show current status
            latest = monitor.get_latest_metrics()
            if latest:
                cpu_avg = np.mean(latest['cpu']['load_per_core'])
                memory_gb = latest['memory']['used_gb']

                gpu_info = ""
                if 'gpu' in latest and latest['gpu']:
                    gpu_mem = list(latest['gpu'].values())[0]['memory_percent']
                    gpu_info = f" | GPU: {gpu_mem:.1f}%"

                print(f"Status: CPU {cpu_avg:.1f}% avg, Mem {memory_gb:.1f}GB{gpu_info}")

        monitor.stop_monitoring()

        # Generate report
        summary = monitor._create_summary()
        print("\n📈 Session Summary:")
        print(f"   Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"   CPU Avg: {summary['cpu']['avg_percent']:.1f}%")
        print(f"   Memory Avg: {summary['memory']['avg_percent']:.1f}%")

        if 'gpu' in summary:
            print(f"   GPU Memory Avg: {summary['gpu']['avg_memory_percent']:.1f}%")

    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
        monitor.stop_monitoring()


def benchmark_system():
    """Run comprehensive system benchmarks."""
    print("🔬 Running System Benchmarks")
    print("=" * 40)

    # CPU benchmark
    print("Testing CPU performance...")
    start = time.time()
    result = sum(i * i for i in range(1000000))
    cpu_time = time.time() - start
    print(f"CPU Benchmark: {cpu_time:.3f}s (sum of squares)")

    # Memory benchmark
    print("Testing memory performance...")
    start = time.time()
    large_array = np.random.random((1000, 1000, 10))  # ~80MB array
    memory_result = np.sum(large_array)
    memory_time = time.time() - start
    print(f"Memory Benchmark: {memory_time:.3f}s ({large_array.nbytes / 1024**2:.1f}MB array)")

    # GPU benchmark (if available)
    try:
        import torch

        if torch.cuda.is_available():
            print("Testing GPU performance...")
            device = torch.device('cuda')

            # Matrix multiplication benchmark
            size = 2000
            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)

            start = time.time()
            for _ in range(5):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) / 5

            print(f"GPU Benchmark: {gpu_time:.3f}s ({size}x{size} matrix mult)")

            # Memory bandwidth test
            start = time.time()
            for _ in range(10):
                _ = torch.randn(1000, 1000, device=device)
            torch.cuda.synchronize()
            memory_bandwidth_time = (time.time() - start) / 10

            print(f"GPU Memory Bandwidth: {memory_bandwidth_time:.3f}s per allocation")

        else:
            print("⚠️  GPU not available for benchmarking")

    except Exception as e:
        print(f"⚠️  GPU benchmarking failed: {e}")

    print("✅ Benchmarks complete")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Performance Monitor for AI Trading System')
    parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in minutes')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks instead of monitoring')
    parser.add_argument('--export-csv', type=str, help='Export metrics to CSV file')

    args = parser.parse_args()

    if args.benchmark:
        benchmark_system()
    else:
        monitor_training_session(args.duration)

        if args.export_csv:
            # Create a temporary monitor to export data
            monitor = PerformanceMonitor()
            monitor.metrics = []  # Would be populated from actual monitoring
            print("⚠️  CSV export requires actual monitoring data")
            print("   Run monitoring first, then use the saved JSON file")


if __name__ == "__main__":
    main()
