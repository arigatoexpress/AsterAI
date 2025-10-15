"""
Test GPU Utilities and Configuration
Tests GPU detection, device selection, and optimizations
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("""
╔════════════════════════════════════════════════════════════════╗
║           Testing GPU Utils & RTX 5070 Ti Configuration        ║
╚════════════════════════════════════════════════════════════════╝
""")

def test_gpu_detection():
    """Test GPU detection and configuration"""
    print(f"\n{'='*70}")
    print("Testing GPU Detection")
    print(f"{'='*70}")

    try:
        from mcp_trader.utils.gpu_utils import get_gpu_manager, gpu_available, get_device

        manager = get_gpu_manager()
        config = manager.config

        print(f"GPU Available: {config.available}")
        print(f"Device Count: {config.device_count}")
        print(f"Device Name: {config.device_name}")
        print(f"Total Memory: {config.total_memory_gb:.1f} GB")
        print(f"CUDA Version: {config.cuda_version}")
        print(f"PyTorch Version: {config.pytorch_version}")
        print(f"Optimal Batch Size: {config.optimal_batch_size}")
        print(f"Supports BF16: {config.supports_bfloat16}")
        print(f"Supports FP16: {config.supports_float16}")

        # Test convenience functions
        print(f"GPU Available (convenience): {gpu_available()}")
        print(f"Device (convenience): {get_device()}")

        # Memory stats
        memory_stats = manager.get_memory_stats()
        print(f"Memory Allocated: {memory_stats['allocated_gb']:.2f} GB")
        print(f"Memory Reserved: {memory_stats['reserved_gb']:.2f} GB")
        print(f"Memory Total: {memory_stats['total_gb']:.2f} GB")
        print(f"Memory Utilization: {memory_stats.get('utilization_pct', 0):.1f}%")

        print("[OK] GPU detection test passed")
        return True

    except Exception as e:
        print(f"[ERROR] GPU detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensor_operations():
    """Test basic tensor operations on detected device"""
    print(f"\n{'='*70}")
    print("Testing Tensor Operations")
    print(f"{'='*70}")

    try:
        from mcp_trader.utils.gpu_utils import get_gpu_manager, move_to_device

        manager = get_gpu_manager()
        device = manager.get_device()

        print(f"Using device: {device}")

        # Test basic tensor creation
        x = torch.randn(10, 10)
        print(f"Created CPU tensor: {x.shape}")

        # Move to device
        x_device = move_to_device(x)
        print(f"Moved to device: {x_device.device}")

        # Test computation
        y = torch.matmul(x_device, x_device.t())
        print(f"Matrix multiplication result: {y.shape}")

        # Test different data types
        if manager.config.available:
            if manager.config.supports_bfloat16:
                x_bf16 = x_device.to(torch.bfloat16)
                print("BF16 conversion: SUCCESS")

            x_fp16 = x_device.to(torch.float16)
            print("FP16 conversion: SUCCESS")

        # Clear cache
        manager.empty_cache()

        print("[OK] Tensor operations test passed")
        return True

    except Exception as e:
        print(f"[ERROR] Tensor operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader():
    """Test optimized DataLoader creation"""
    print(f"\n{'='*70}")
    print("Testing DataLoader Creation")
    print(f"{'='*70}")

    try:
        from mcp_trader.utils.gpu_utils import get_gpu_manager
        from torch.utils.data import TensorDataset

        manager = get_gpu_manager()

        # Create dummy dataset
        x = torch.randn(1000, 50)
        y = torch.randn(1000, 1)
        dataset = TensorDataset(x, y)

        # Test DataLoader creation
        data_loader = manager.create_data_loader(dataset, batch_size=32)

        print(f"DataLoader created with batch_size: {data_loader.batch_size}")
        print(f"Number of workers: {data_loader.num_workers}")
        print(f"Pin memory: {data_loader.pin_memory}")

        # Test iteration
        for batch_x, batch_y in data_loader:
            print(f"Batch shapes: {batch_x.shape}, {batch_y.shape}")
            break

        print("[OK] DataLoader test passed")
        return True

    except Exception as e:
        print(f"[ERROR] DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_precision():
    """Test mixed precision setup"""
    print(f"\n{'='*70}")
    print("Testing Mixed Precision Setup")
    print(f"{'='*70}")

    try:
        from mcp_trader.utils.gpu_utils import get_gpu_manager

        manager = get_gpu_manager()

        if not manager.config.available:
            print("[SKIP] Mixed precision test skipped (CPU mode)")
            return True

        # Test mixed precision setup
        autocast_ctx, scaler, cleanup = manager.setup_automatic_mixed_precision()

        if autocast_ctx is None:
            print("[SKIP] Mixed precision not available")
            return True

        print("Mixed precision setup: SUCCESS")
        print(f"Optimal dtype: {manager.get_optimal_dtype()}")

        # Test with a simple model
        model = torch.nn.Linear(10, 1).to(manager.get_device())
        optimizer = manager.create_optimizer(model)

        x = torch.randn(5, 10).to(manager.get_device())
        y = torch.randn(5, 1).to(manager.get_device())

        with autocast_ctx:
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print("Mixed precision training step: SUCCESS")

        cleanup()
        print("[OK] Mixed precision test passed")
        return True

    except Exception as e:
        print(f"[ERROR] Mixed precision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark():
    """Test GPU benchmarking"""
    print(f"\n{'='*70}")
    print("Testing GPU Benchmark")
    print(f"{'='*70}")

    try:
        from mcp_trader.utils.gpu_utils import get_gpu_manager

        manager = get_gpu_manager()

        if not manager.config.available:
            print("[SKIP] Benchmark test skipped (CPU mode)")
            return True

        # Create simple model for benchmarking
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        ).to(manager.get_device())

        # Benchmark
        results = manager.benchmark_gpu(model, (32, 100))

        print(f"Inference time: {results['inference_ms']:.2f} ms")
        print(f"Throughput: {results['throughput']:.1f} samples/sec")
        print(f"Device: {results['device']}")
        print(f"Input shape: {results['input_shape']}")

        print("[OK] Benchmark test passed")
        return True

    except Exception as e:
        print(f"[ERROR] Benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__ if 'torch' in sys.modules else 'Not imported'}")

    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Tensor Operations", test_tensor_operations),
        ("DataLoader Creation", test_data_loader),
        ("Mixed Precision", test_mixed_precision),
        ("GPU Benchmark", test_benchmark)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your GPU setup is working perfectly!")
        print("\nNext: Try the data pipeline and start collecting historical data")
    else:
        print(f"\n⚠️ Some tests failed. Check the output above for details.")
        print("You can still use CPU fallback mode for most operations")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()



