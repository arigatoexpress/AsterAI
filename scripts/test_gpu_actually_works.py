"""
Test if RTX 5070 Ti actually works with PyTorch despite sm_120 warning
"""
import torch
import warnings
import time

warnings.filterwarnings('ignore')

print("="*70)
print("RTX 5070 Ti GPU Functionality Test")
print("="*70)

# Basic info
print(f"\n✓ PyTorch Version: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")

# Test 1: Simple tensor operations
print("\n" + "="*70)
print("Test 1: Basic Tensor Operations")
print("="*70)
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ Matrix multiplication: PASSED")
except Exception as e:
    print(f"✗ Matrix multiplication: FAILED - {e}")
    exit(1)

# Test 2: Neural network operations
print("\n" + "="*70)
print("Test 2: Neural Network Operations")
print("="*70)
try:
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).cuda()
    
    x = torch.randn(32, 100).cuda()
    output = model(x)
    print(f"✓ Forward pass: PASSED (output shape: {output.shape})")
except Exception as e:
    print(f"✗ Forward pass: FAILED - {e}")
    exit(1)

# Test 3: LSTM (critical for trading)
print("\n" + "="*70)
print("Test 3: LSTM Operations (Critical for Trading)")
print("="*70)
try:
    lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True).cuda()
    x = torch.randn(32, 50, 10).cuda()  # batch=32, seq_len=50, features=10
    output, (h_n, c_n) = lstm(x)
    print(f"✓ LSTM forward pass: PASSED (output shape: {output.shape})")
except Exception as e:
    print(f"✗ LSTM forward pass: FAILED - {e}")
    exit(1)

# Test 4: Gradient computation
print("\n" + "="*70)
print("Test 4: Backpropagation")
print("="*70)
try:
    x = torch.randn(10, 10, requires_grad=True).cuda()
    y = (x ** 2).sum()
    y.backward()
    print(f"✓ Gradient computation: PASSED (grad shape: {x.grad.shape})")
except Exception as e:
    print(f"✗ Gradient computation: FAILED - {e}")
    exit(1)

# Test 5: Performance benchmark
print("\n" + "="*70)
print("Test 5: Performance Benchmark (GPU vs CPU)")
print("="*70)
size = 2000
iterations = 50

# GPU benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(iterations):
    x = torch.randn(size, size).cuda()
    y = torch.randn(size, size).cuda()
    z = torch.matmul(x, y)
torch.cuda.synchronize()
gpu_time = time.time() - start

# CPU benchmark
start = time.time()
for _ in range(iterations):
    x = torch.randn(size, size)
    y = torch.randn(size, size)
    z = torch.matmul(x, y)
cpu_time = time.time() - start

speedup = cpu_time / gpu_time
print(f"✓ GPU Time: {gpu_time:.3f}s")
print(f"✓ CPU Time: {cpu_time:.3f}s")
print(f"✓ Speedup: {speedup:.1f}x")

# Final verdict
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)
if speedup > 2:
    print(f"✅ GPU is {speedup:.1f}x faster than CPU - FULLY FUNCTIONAL")
    print("✅ You can use this GPU for training despite the sm_120 warning!")
    print("✅ The warning is about optimal kernel support, but PyTorch falls back")
    print("   to compatible kernels and still provides significant acceleration.")
    exit(0)
else:
    print(f"⚠️ GPU is only {speedup:.1f}x faster - limited acceleration")
    print("⚠️ Consider CPU training or wait for PyTorch sm_120 support")
    exit(1)

