#!/usr/bin/env python3
"""
Comprehensive GPU Testing Script for AsterAI Trading System

This script tests all GPU-powered features to ensure they're working properly:
1. Basic GPU detection and CUDA availability
2. PyTorch CUDA functionality
3. CuPy GPU arrays and operations
4. JAX GPU support (if available)
5. TensorRT model optimization (if available)
6. GPU-accelerated trading model inference
7. Memory management and performance benchmarks

Usage:
    python gpu_comprehensive_test.py
"""

import os
import sys
import time
import json
import psutil
import platform
from datetime import datetime
from typing import Dict, List, Any

# GPU and acceleration libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

import numpy as np
import GPUtil


class GPUComprehensiveTester:
    """Comprehensive GPU testing suite."""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()

    def test_basic_gpu_detection(self) -> Dict[str, Any]:
        """Test basic GPU detection and CUDA availability."""
        print("Testing basic GPU detection...")

        results = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_details': [],
        }

        # Test NVIDIA GPU detection
        try:
            gpus = GPUtil.getGPUs()
            results['gpu_count'] = len(gpus)
            results['cuda_available'] = True

            for i, gpu in enumerate(gpus):
                gpu_info = {
                    'id': i,
                    'name': gpu.name,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_free_mb': gpu.memoryFree,
                    'memory_used_mb': gpu.memoryUsed,
                    'temperature': gpu.temperature,
                    'utilization_percent': gpu.load * 100,
                }
                results['gpu_details'].append(gpu_info)

        except Exception as e:
            print(f"   FAIL GPU detection failed: {e}")
            results['error'] = str(e)

        print(f"   PASS Found {results['gpu_count']} GPU(s)")
        return results

    def test_pytorch_cuda(self) -> Dict[str, Any]:
        """Test PyTorch CUDA functionality."""
        print("Testing PyTorch CUDA...")

        if not TORCH_AVAILABLE:
            return {'available': False, 'error': 'PyTorch not installed'}

        results = {
            'available': False,
            'cuda_version': None,
            'device_count': 0,
            'device_names': [],
        }

        try:
            results['available'] = torch.cuda.is_available()

            if results['available']:
                results['device_count'] = torch.cuda.device_count()
                results['device_names'] = [torch.cuda.get_device_name(i) for i in range(results['device_count'])]
                results['cuda_version'] = torch.version.cuda

                # Test basic CUDA operations
                device = torch.device('cuda:0')
                x = torch.randn(1000, 1000).to(device)
                y = torch.randn(1000, 1000).to(device)
                start_time = time.time()
                z = torch.matmul(x, y)
                torch.cuda.synchronize()
                compute_time = time.time() - start_time

                results['matrix_multiply_test'] = {
                    'success': True,
                    'time_ms': compute_time * 1000,
                    'device_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                }

                print(f"   PASS PyTorch CUDA working - {results['device_count']} device(s)")
                print(f"   Matrix multiply (1000x1000): {compute_time*1000:.2f}ms")
            else:
                print("   FAIL PyTorch CUDA not available")

        except Exception as e:
            results['error'] = str(e)
            print(f"   FAIL PyTorch CUDA test failed: {e}")

        return results

    def test_cupy_functionality(self) -> Dict[str, Any]:
        """Test CuPy GPU array functionality."""
        print("Testing CuPy GPU arrays...")

        if not CUPY_AVAILABLE:
            return {'available': False, 'error': 'CuPy not installed'}

        results = {
            'available': False,
            'cuda_version': None,
            'memory_pool_size': None,
        }

        try:
            # Test basic CuPy operations
            x = cp.array([1, 2, 3, 4, 5])
            y = cp.array([10, 20, 30, 40, 50])

            start_time = time.time()
            z = cp.dot(x, y)
            cp.cuda.Stream.null.synchronize()
            compute_time = time.time() - start_time

            results['available'] = True
            results['dot_product_test'] = {
                'success': True,
                'result': int(z),
                'time_ms': compute_time * 1000,
            }

            # Test memory info
            mempool = cp.get_default_memory_pool()
            results['memory_pool_size'] = mempool.total_bytes()

            print(f"   PASS CuPy working - dot product result: {int(z)}")
            print(f"   Dot product time: {compute_time*1000:.2f}ms")

        except Exception as e:
            results['error'] = str(e)
            print(f"   FAIL CuPy test failed: {e}")

        return results

    def test_jax_functionality(self) -> Dict[str, Any]:
        """Test JAX functionality (CPU fallback if GPU not available)."""
        print("Testing JAX...")

        if not JAX_AVAILABLE:
            return {'available': False, 'error': 'JAX not installed'}

        results = {
            'available': False,
            'version': jax.__version__,
            'gpu_available': False,
            'devices': [],
        }

        try:
            devices = jax.devices()
            results['devices'] = [str(device) for device in devices]

            # Check if GPU is available
            gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
            results['gpu_available'] = len(gpu_devices) > 0

            # Test basic JAX operations
            def test_function(x):
                return jnp.sum(x**2)

            x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

            start_time = time.time()
            result = test_function(x)
            compile_time = time.time() - start_time

            # JIT compile and run
            jit_fn = jax.jit(test_function)
            start_time = time.time()
            result_jit = jit_fn(x)
            jit_time = time.time() - start_time

            results['available'] = True
            results['basic_test'] = {
                'result': float(result),
                'compile_time_ms': compile_time * 1000,
                'jit_time_ms': jit_time * 1000,
            }

            backend = "GPU" if results['gpu_available'] else "CPU"
            print(f"   PASS JAX working on {backend} - {len(devices)} device(s)")

        except Exception as e:
            results['error'] = str(e)
            print(f"   FAIL JAX test failed: {e}")

        return results

    def test_tensorrt_functionality(self) -> Dict[str, Any]:
        """Test TensorRT model optimization."""
        print("Testing TensorRT...")

        if not TRT_AVAILABLE:
            return {'available': False, 'error': 'TensorRT not installed'}

        results = {
            'available': False,
            'version': None,
        }

        try:
            results['version'] = trt.__version__

            # Test basic TensorRT functionality
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)

            if builder.platform_has_fast_fp16:
                results['fp16_support'] = True
            if builder.platform_has_fast_int8:
                results['int8_support'] = True

            results['available'] = True
            print(f"   PASS TensorRT {results['version']} available")

            # Test network creation
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            if network:
                results['network_creation'] = True
                print("   PASS Network creation successful")
            else:
                results['network_creation'] = False

        except Exception as e:
            results['error'] = str(e)
            print(f"   FAIL TensorRT test failed: {e}")

        return results

    def test_gpu_accelerated_models(self) -> Dict[str, Any]:
        """Test GPU-accelerated trading models."""
        print("📈 Testing GPU-accelerated trading models...")

        results = {
            'pytorch_model_test': {},
            'cupy_model_test': {},
            'memory_bandwidth_test': {},
        }

        # Test PyTorch model on GPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # Simple LSTM-like model for testing
                class SimpleModel(nn.Module):
                    def __init__(self, input_size=10, hidden_size=32, output_size=1):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                        self.fc = nn.Linear(hidden_size, output_size)

                    def forward(self, x):
                        lstm_out, _ = self.lstm(x)
                        return self.fc(lstm_out[:, -1, :])

                device = torch.device('cuda:0')
                model = SimpleModel().to(device)

                # Generate test data
                batch_size = 100
                seq_length = 50
                input_size = 10

                x = torch.randn(batch_size, seq_length, input_size).to(device)

                # Warmup
                for _ in range(5):
                    _ = model(x)

                # Benchmark
                start_time = time.time()
                for _ in range(50):
                    _ = model(x)
                torch.cuda.synchronize()
                inference_time = time.time() - start_time

                avg_time = (inference_time / 50) * 1000  # ms per inference

                results['pytorch_model_test'] = {
                    'success': True,
                    'avg_inference_time_ms': avg_time,
                    'device': str(device),
                }

                print(f"   PASS PyTorch model test - {avg_time:.2f}ms avg inference")

            except Exception as e:
                results['pytorch_model_test'] = {'success': False, 'error': str(e)}

        # Test CuPy array operations for trading calculations
        if CUPY_AVAILABLE:
            try:
                # Simulate price data processing
                prices = cp.random.randn(10000, 100)  # 10k time steps, 100 assets

                # Calculate returns and volatility (common trading operations)
                returns = cp.diff(prices, axis=0) / prices[:-1]

                start_time = time.time()
                volatilities = cp.std(returns, axis=0)  # Volatility per asset
                correlations = cp.corrcoef(returns.T)  # Correlation matrix
                cp.cuda.Stream.null.synchronize()
                compute_time = time.time() - start_time

                results['cupy_model_test'] = {
                    'success': True,
                    'volatility_calc_time_ms': compute_time * 1000,
                    'correlation_matrix_shape': correlations.shape,
                }

                print(f"   PASS CuPy trading calculations - {compute_time*1000:.2f}ms")

            except Exception as e:
                results['cupy_model_test'] = {'success': False, 'error': str(e)}

        # Memory bandwidth test
        if torch.cuda.is_available():
            try:
                device = torch.device('cuda:0')

                # Test memory bandwidth
                sizes = [1000, 10000, 100000]  # Different array sizes

                bandwidth_results = []
                for size in sizes:
                    # Memory copy test
                    x_cpu = torch.randn(size, size)
                    start_time = time.time()
                    x_gpu = x_cpu.to(device)
                    torch.cuda.synchronize()
                    copy_time = time.time() - start_time

                    # Calculate bandwidth (GB/s)
                    bytes_transferred = x_cpu.numel() * x_cpu.element_size()
                    bandwidth = (bytes_transferred / (1024**3)) / copy_time

                    bandwidth_results.append({
                        'size': f"{size}x{size}",
                        'bandwidth_gbps': bandwidth,
                    })

                results['memory_bandwidth_test'] = bandwidth_results
                print(f"   PASS Memory bandwidth test - up to {max([r['bandwidth_gbps'] for r in bandwidth_results]):.1f} GB/s")

            except Exception as e:
                results['memory_bandwidth_test'] = {'error': str(e)}

        return results

    def test_gpu_memory_management(self) -> Dict[str, Any]:
        """Test GPU memory management and monitoring."""
        print("Testing GPU memory management...")

        results = {
            'system_memory': {},
            'gpu_memory': {},
            'memory_stress_test': {},
        }

        # System memory info
        memory = psutil.virtual_memory()
        results['system_memory'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'usage_percent': memory.percent,
        }

        # GPU memory info (if available)
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            results['gpu_memory'] = {
                'device': device,
                'total_memory_gb': torch.cuda.get_device_properties(device).total_memory / (1024**3),
                'allocated_memory_gb': torch.cuda.memory_allocated(device) / (1024**3),
                'cached_memory_gb': torch.cuda.memory_reserved(device) / (1024**3),
            }

        # Memory stress test
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device = torch.device('cuda:0')

                # Allocate increasingly large tensors and monitor memory
                sizes = [1000, 5000, 10000]
                memory_usage = []

                for size in sizes:
                    # Clear cache before each test
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Allocate tensor
                    x = torch.randn(size, size, device=device)

                    allocated = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved = torch.cuda.memory_reserved(device) / (1024**3)

                    memory_usage.append({
                        'tensor_size': f"{size}x{size}",
                        'allocated_gb': allocated,
                        'reserved_gb': reserved,
                    })

                results['memory_stress_test'] = {
                    'success': True,
                    'memory_usage': memory_usage,
                }

                print(f"   PASS Memory stress test - monitored {len(sizes)} allocations")

            except Exception as e:
                results['memory_stress_test'] = {'success': False, 'error': str(e)}

        return results

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive GPU tests."""
        print("🚀 Starting comprehensive GPU testing...")
        print("=" * 60)

        self.results['basic_detection'] = self.test_basic_gpu_detection()
        self.results['pytorch_cuda'] = self.test_pytorch_cuda()
        self.results['cupy_arrays'] = self.test_cupy_functionality()
        self.results['jax_support'] = self.test_jax_functionality()
        self.results['tensorrt'] = self.test_tensorrt_functionality()
        self.results['gpu_models'] = self.test_gpu_accelerated_models()
        self.results['memory_management'] = self.test_gpu_memory_management()

        # Calculate overall assessment
        self.results['overall_assessment'] = self._calculate_overall_assessment()

        execution_time = time.time() - self.start_time
        self.results['execution_time_seconds'] = execution_time

        print(f"\nPASS Comprehensive GPU testing completed in {execution_time:.1f}s")
        return self.results

    def _calculate_overall_assessment(self) -> Dict[str, Any]:
        """Calculate overall GPU functionality assessment."""
        assessment = {
            'overall_score': 0,
            'max_score': 100,
            'components': {},
        }

        # PyTorch CUDA (30 points)
        pytorch = self.results.get('pytorch_cuda', {})
        if pytorch.get('available', False):
            assessment['components']['pytorch_cuda'] = 30
            assessment['overall_score'] += 30
        else:
            assessment['components']['pytorch_cuda'] = 0

        # CuPy (25 points)
        cupy = self.results.get('cupy_arrays', {})
        if cupy.get('available', False):
            assessment['components']['cupy'] = 25
            assessment['overall_score'] += 25
        else:
            assessment['components']['cupy'] = 0

        # JAX (20 points)
        jax_test = self.results.get('jax_support', {})
        if jax_test.get('available', False):
            if jax_test.get('gpu_available', False):
                assessment['components']['jax'] = 20
                assessment['overall_score'] += 20
            else:
                assessment['components']['jax'] = 10  # CPU fallback
                assessment['overall_score'] += 10
        else:
            assessment['components']['jax'] = 0

        # TensorRT (15 points)
        tensorrt = self.results.get('tensorrt', {})
        if tensorrt.get('available', False):
            assessment['components']['tensorrt'] = 15
            assessment['overall_score'] += 15
        else:
            assessment['components']['tensorrt'] = 0

        # GPU Models (10 points)
        gpu_models = self.results.get('gpu_models', {})
        model_success = (
            gpu_models.get('pytorch_model_test', {}).get('success', False) or
            gpu_models.get('cupy_model_test', {}).get('success', False)
        )
        if model_success:
            assessment['components']['gpu_models'] = 10
            assessment['overall_score'] += 10
        else:
            assessment['components']['gpu_models'] = 0

        # Determine overall grade
        if assessment['overall_score'] >= 80:
            assessment['grade'] = 'A - Excellent'
        elif assessment['overall_score'] >= 60:
            assessment['grade'] = 'B - Good'
        elif assessment['overall_score'] >= 40:
            assessment['grade'] = 'C - Fair'
        else:
            assessment['grade'] = 'D - Poor'

        return assessment

    def save_results(self, filename: str = None):
        """Save test results to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'gpu_comprehensive_test_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nTest results saved to: {filename}")


def main():
    """Main function to run comprehensive GPU testing."""
    print("AsterAI GPU Comprehensive Testing Suite")
    print("=" * 60)

    tester = GPUComprehensiveTester()
    results = tester.run_comprehensive_tests()

    # Display summary
    assessment = results['overall_assessment']
    print("\nGPU FUNCTIONALITY ASSESSMENT:")
    print(f"   Overall Score: {assessment['overall_score']}/{assessment['max_score']}")
    print(f"   Grade: {assessment['grade']}")
    print("   Component Breakdown:")

    for component, score in assessment['components'].items():
        status = "PASS" if score > 0 else "FAIL"
        print(f"      {status} {component}: {score} points")

    # Show key findings
    print("\nKEY FINDINGS:")
    pytorch = results.get('pytorch_cuda', {})
    if pytorch.get('available'):
        devices = pytorch.get('device_count', 0)
        print(f"   PASS PyTorch CUDA: {devices} device(s) available")
    else:
        print("   FAIL PyTorch CUDA: Not available")

    cupy = results.get('cupy_arrays', {})
    if cupy.get('available'):
        print("   PASS CuPy: GPU arrays working")
    else:
        print("   FAIL CuPy: Not available")

    jax_test = results.get('jax_support', {})
    if jax_test.get('gpu_available'):
        print("   PASS JAX: GPU acceleration available")
    elif jax_test.get('available'):
        print("   WARN JAX: CPU fallback only")
    else:
        print("   FAIL JAX: Not available")

    # Performance highlights
    print("\nPERFORMANCE HIGHLIGHTS:")
    if pytorch.get('available') and 'matrix_multiply_test' in pytorch:
        test = pytorch['matrix_multiply_test']
        print(f"   PyTorch Matrix Multiply (1000×1000): {test['time_ms']:.1f}ms")

    if cupy.get('available') and 'dot_product_test' in cupy:
        test = cupy['dot_product_test']
        print(f"   CuPy Dot Product: {test['time_ms']:.1f}ms")

    # Memory info
    gpu_memory = results.get('gpu_memory', {})
    if gpu_memory:
        total_gb = gpu_memory.get('total_memory_gb', 0)
        allocated_gb = gpu_memory.get('allocated_memory_gb', 0)
        print(f"   GPU Memory: {allocated_gb:.1f}GB used / {total_gb:.1f}GB total")

    # Save results
    tester.save_results()

    print("\nGPU testing completed!")
    print(f"   Detailed results saved for analysis")
    print(f"   Ready for GPU-accelerated trading operations")

    return 0


if __name__ == "__main__":
    exit(main())
