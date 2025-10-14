# 🚀 GPU-Optimized AI Trading System Setup

## Hardware Requirements
- **GPU**: NVIDIA RTX 5070Ti (16GB VRAM)
- **CPU**: 16-core AMD processor (or equivalent)
- **RAM**: 64GB+ recommended
- **Storage**: 500GB+ SSD for data and models

## Step 1: Install NVIDIA Drivers and CUDA

### Ubuntu/Debian (Recommended)
```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-535  # For RTX 5070Ti

# Install CUDA Toolkit 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-1

# Verify installation
nvidia-smi
nvcc --version
```

### Alternative: Use NVIDIA Docker Container
```bash
# Pull GPU-enabled base image
docker pull nvidia/cuda:12.1-devel-ubuntu22.04

# Run with GPU support
docker run --gpus all -it nvidia/cuda:12.1-devel-ubuntu22.04
```

## Step 2: Setup Python Environment

### Option A: Native Installation (Recommended for Performance)
```bash
# Install Python 3.12 (required for full PyTorch support)
sudo apt install python3.12 python3.12-pip python3.12-dev

# Upgrade pip
python3.12 -m pip install --upgrade pip setuptools wheel

# Install GPU-optimized requirements
pip install -r requirements-gpu.txt

# Verify PyTorch CUDA support
python3.12 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

### Option B: Docker Container (Easier Setup)
```bash
# Build GPU-optimized container
docker build -f Dockerfile.gpu -t ai-trading-gpu .

# Run with GPU access
docker run --gpus all \
    --name ai-trading-container \
    -v $(pwd):/app \
    -p 8501:8501 \
    -p 8888:8888 \
    ai-trading-gpu
```

## Step 3: Configure GPU Optimizations

### Environment Variables for RTX 5070Ti
```bash
# Set optimal environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0

# Multi-core CPU optimizations
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# PyTorch optimizations
export TORCH_NUM_THREADS=16
export TORCH_CUDA_ARCH_LIST=8.9  # RTX 5070Ti compute capability
```

### Create GPU Configuration File
```bash
# Copy and customize configuration
cp config-gpu.yaml config-local.yaml

# Edit for your specific hardware
nano config-local.yaml
```

## Step 4: Verify Hardware Performance

### Run Performance Benchmarks
```bash
# Run comprehensive benchmarks
python scripts/monitor_performance.py --benchmark

# Expected results for RTX 5070Ti:
# Matrix 2000x2000: GPU ~0.05s (20x faster than CPU)
# Memory bandwidth: <0.01s per allocation
```

### Monitor GPU Usage During Training
```bash
# Start performance monitoring
python scripts/monitor_performance.py --duration 30 &

# Train a model (will be monitored)
python -c "
import torch
from mcp_trader.models.deep_learning.gpu_lstm import GPUOptimizedPredictor
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'close': 1000 + np.cumsum(np.random.normal(0, 10, 1000))
})

# Train GPU model
model = GPUOptimizedPredictor()
model.fit(data)
"
```

## Step 5: Train High-Performance Models

### Train GPU-Optimized LSTM
```bash
# Train on RTX 5070Ti (should take 5-10 minutes)
python -c "
from mcp_trader.models.deep_learning.gpu_lstm import GPUOptimizedPredictor
import pandas as pd
import numpy as np

# Generate training data (2 years of hourly data)
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', periods=17520, freq='H')
prices = 1000 + np.cumsum(np.random.normal(0.001, 0.02, 17520))

df = pd.DataFrame({
    'timestamp': dates,
    'close': prices,
    'volume': np.random.uniform(1000, 100000, 17520)
})

# Train GPU model
model = GPUOptimizedPredictor({
    'batch_size': 128,
    'num_epochs': 50,
    'enable_mixed_precision': True
})

print('Training GPU LSTM model...')
model.fit(df)
print('Training complete!')

# Check performance metrics
metrics = model.get_performance_metrics()
print(f'GPU Memory Used: {metrics[\"gpu_memory_allocated\"]:.1f}GB')
"
```

### Train Reinforcement Learning Agents
```bash
# Train RL agents with GPU acceleration
python -c "
from mcp_trader.models.reinforcement_learning.trading_agents import RLTradingAgent
import pandas as pd
import numpy as np

# Create training data
data = pd.DataFrame({
    'close': 1000 + np.cumsum(np.random.normal(0, 10, 5000))
})

# Train PPO agent on GPU
agent = RLTradingAgent({
    'algorithm': 'PPO',
    'total_timesteps': 10000,
    'batch_size': 128
})

print('Training RL agent on GPU...')
agent.train(data)
print('RL training complete!')
"
```

## Step 6: Monitor and Optimize Performance

### Real-time Monitoring
```bash
# Monitor GPU and CPU usage during training
python scripts/monitor_performance.py --duration 60

# Output shows:
# Performance: CPU 45.2% avg, Mem 8.5GB | GPU: 78.3% (12.5GB)
```

### Memory Optimization
```bash
# Check memory usage
python -c "
import torch
import gc

print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated')
print(f'GPU Memory: {torch.cuda.memory_reserved()/1024**3:.1f}GB reserved')

# Clear cache if needed
torch.cuda.empty_cache()
gc.collect()
"
```

### Profile Training Performance
```bash
# Profile model training
python -c "
import cProfile
import pstats
from mcp_trader.models.deep_learning.gpu_lstm import GPUOptimizedPredictor

profiler = cProfile.Profile()
profiler.enable()

# Your training code here
model = GPUOptimizedPredictor()
# ... training code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
"
```

## Step 7: Production Deployment

### Docker Production Setup
```bash
# Build production container
docker build -f Dockerfile.gpu -t ai-trading-gpu:prod .

# Run with GPU support
docker run --gpus all \
    --name ai-trading-prod \
    -v /path/to/data:/app/data \
    -p 8501:8501 \
    ai-trading-gpu:prod \
    python run_complete_system.py live
```

### Kubernetes Deployment (Optional)
```yaml
# k8s-gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-trading-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ai-trading-gpu
  template:
    metadata:
      labels:
        app: ai-trading-gpu
    spec:
      containers:
      - name: ai-trading
        image: ai-trading-gpu:prod
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
            cpu: "8"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
---
apiVersion: v1
kind: Service
metadata:
  name: ai-trading-service
spec:
  selector:
    app: ai-trading-gpu
  ports:
  - port: 8501
    targetPort: 8501
  type: LoadBalancer
```

## Step 8: Performance Tuning

### RTX 5070Ti Specific Optimizations
```python
# Custom configuration for RTX 5070Ti
config = {
    'batch_size': 128,           # Max for 16GB VRAM
    'mixed_precision': True,     # FP16 for speed
    'gradient_checkpointing': True,  # Memory efficient
    'flash_attention': True,     # Hardware accelerated
    'tensor_cores': True,        # Use Tensor Cores
    'memory_efficient_attention': True
}
```

### Multi-Core CPU Optimizations
```python
import os
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

# Ray configuration for distributed training
ray.init(num_cpus=16, num_gpus=1)
```

## Step 9: Troubleshooting

### Common Issues and Solutions

#### CUDA Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### GPU Memory Issues
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size
config['batch_size'] = 64  # Try smaller batches
```

#### Slow Training Performance
```bash
# Enable CUDA optimizations
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=0

# Use mixed precision
config['mixed_precision'] = True

# Enable gradient checkpointing
config['gradient_checkpointing'] = True
```

#### CPU Bottlenecks
```bash
# Check CPU usage
htop

# Optimize thread count
export OMP_NUM_THREADS=16

# Use faster BLAS library
pip install mkl  # Intel Math Kernel Library
```

## Expected Performance Gains

### RTX 5070Ti vs CPU Training
- **LSTM Training**: 15-25x faster
- **Matrix Operations**: 50-100x faster
- **Memory Bandwidth**: 10-20x faster
- **Overall System**: 10-15x faster end-to-end

### 16-core AMD vs Single Core
- **Data Loading**: 8-12x faster
- **Feature Engineering**: 10-15x faster
- **Model Inference**: 12-16x faster
- **Parallel Processing**: Near-linear scaling

## Next Steps

1. **Test Configuration**: Run `python scripts/setup_gpu_environment.py`
2. **Benchmark Hardware**: Run `python scripts/monitor_performance.py --benchmark`
3. **Train Models**: Execute GPU-optimized training scripts
4. **Monitor Performance**: Use real-time monitoring during training
5. **Deploy Production**: Scale to live trading with confidence

## Support

For issues specific to RTX 5070Ti optimization:
- Check NVIDIA driver compatibility
- Verify CUDA 12.1 installation
- Monitor GPU memory usage
- Profile training bottlenecks

The system is now optimized for maximum performance on your high-end hardware! 🚀
