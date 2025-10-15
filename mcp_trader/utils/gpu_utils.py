"""
GPU Utilities for RTX 5070 Ti Support with CPU Fallback
Provides device detection, memory management, and GPU-aware training
"""

import os
import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GPUConfig:
    """GPU configuration and capabilities"""
    available: bool = False
    device_count: int = 0
    current_device: int = 0
    device_name: str = "CPU"
    total_memory_gb: float = 0.0
    cuda_version: Optional[str] = None
    pytorch_version: str = "unknown"
    supports_bfloat16: bool = False
    supports_float16: bool = True
    optimal_batch_size: int = 32
    memory_fraction: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            'available': self.available,
            'device_count': self.device_count,
            'current_device': self.current_device,
            'device_name': self.device_name,
            'total_memory_gb': self.total_memory_gb,
            'cuda_version': self.cuda_version,
            'pytorch_version': self.pytorch_version,
            'supports_bfloat16': self.supports_bfloat16,
            'supports_float16': self.supports_float16,
            'optimal_batch_size': self.optimal_batch_size,
            'memory_fraction': self.memory_fraction
        }


class GPUManager:
    """
    Manages GPU resources with automatic fallback to CPU
    Optimized for RTX 5070 Ti and similar high-end GPUs
    """

    def __init__(self):
        self.config = self._detect_gpu()
        self._setup_gpu_optimizations()
        logger.info(f"GPU Manager initialized: {self.config.device_name}")

    def _detect_gpu(self) -> GPUConfig:
        """Detect GPU capabilities with fallback"""
        config = GPUConfig()

        try:
            # Check PyTorch installation
            config.pytorch_version = torch.__version__

            # Check CUDA availability
            if torch.cuda.is_available():
                config.available = True
                config.device_count = torch.cuda.device_count()
                config.current_device = torch.cuda.current_device()

                # Get device info
                device_props = torch.cuda.get_device_properties(config.current_device)
                config.device_name = device_props.name
                config.total_memory_gb = device_props.total_memory / (1024**3)

                # CUDA version
                config.cuda_version = torch.version.cuda

                # Data type support (Ada Lovelace features)
                config.supports_bfloat16 = True  # RTX 30/40 series support
                config.supports_float16 = True

                # Optimal batch sizes based on GPU memory
                if config.total_memory_gb >= 16:  # RTX 4070 Ti, 4080, 4090
                    config.optimal_batch_size = 512
                elif config.total_memory_gb >= 12:  # RTX 4070, 4080
                    config.optimal_batch_size = 256
                elif config.total_memory_gb >= 8:  # RTX 3060 Ti, 3070, 4060 Ti
                    config.optimal_batch_size = 128
                else:
                    config.optimal_batch_size = 64

                # Memory fraction (leave some for system)
                config.memory_fraction = 0.85 if config.total_memory_gb >= 16 else 0.9

                logger.info(f"✅ GPU detected: {config.device_name} ({config.total_memory_gb:.1f}GB)")
                logger.info(f"   CUDA: {config.cuda_version}, PyTorch: {config.pytorch_version}")
                logger.info(f"   Optimal batch size: {config.optimal_batch_size}")

            else:
                logger.warning("⚠️ CUDA not available, using CPU fallback")
                config.device_name = "CPU (Fallback)"
                config.optimal_batch_size = 32

        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            logger.warning("Using CPU fallback due to GPU detection error")
            config.device_name = "CPU (Error Fallback)"
            config.optimal_batch_size = 16

        return config

    def _setup_gpu_optimizations(self):
        """Setup GPU optimizations for RTX 30/40 series"""
        if not self.config.available:
            return

        try:
            # Enable TF32 for faster matrix operations (Ada Lovelace feature)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.debug("✓ TF32 enabled for faster matrix operations")

            # Enable cuDNN benchmark for optimal performance
            torch.backends.cudnn.benchmark = True
            logger.debug("✓ cuDNN benchmark enabled")

            # Set memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
            logger.debug(f"✓ Memory fraction set to {self.config.memory_fraction}")

            # Enable async data loading if supported
            if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'set_device'):
                torch.cuda.set_device(self.config.current_device)

        except Exception as e:
            logger.warning(f"Some GPU optimizations failed: {e}")

    def get_device(self) -> torch.device:
        """Get the best available device"""
        if self.config.available:
            return torch.device(f'cuda:{self.config.current_device}')
        else:
            return torch.device('cpu')

    def get_optimal_dtype(self) -> torch.dtype:
        """Get optimal data type for training"""
        if self.config.available and self.config.supports_bfloat16:
            return torch.bfloat16  # Better for Ada Lovelace
        elif self.config.available and self.config.supports_float16:
            return torch.float16
        else:
            return torch.float32

    def empty_cache(self):
        """Empty GPU cache if available"""
        if self.config.available:
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if not self.config.available:
            return {'allocated_gb': 0.0, 'reserved_gb': 0.0, 'total_gb': 0.0}

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'utilization_pct': (allocated / total) * 100
        }

    def move_to_device(self, data: Any) -> Any:
        """Move data to appropriate device"""
        device = self.get_device()
        if hasattr(data, 'to'):
            return data.to(device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self.move_to_device(item) for item in data)
        elif isinstance(data, dict):
            return {key: self.move_to_device(value) for key, value in data.items()}
        else:
            return data

    def create_data_loader(self, dataset, batch_size: Optional[int] = None, **kwargs):
        """Create optimized DataLoader"""
        from torch.utils.data import DataLoader

        if batch_size is None:
            batch_size = self.config.optimal_batch_size

        # Adjust batch size based on available memory
        memory_stats = self.get_memory_stats()
        if memory_stats['utilization_pct'] > 80:
            batch_size = max(1, batch_size // 2)
            logger.warning(f"High memory usage, reducing batch size to {batch_size}")

        default_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 4 if self.config.available else 0,  # GPU: use workers, CPU: no workers
            'pin_memory': self.config.available,  # Faster CPU->GPU transfer
            'persistent_workers': True,
            'prefetch_factor': 2
        }

        # Override defaults with user kwargs
        default_kwargs.update(kwargs)

        return DataLoader(dataset, **default_kwargs)

    def setup_automatic_mixed_precision(self):
        """Setup automatic mixed precision training"""
        if not self.config.available:
            return None, None, lambda: None

        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            autocast_context = autocast(dtype=self.get_optimal_dtype())

            def cleanup():
                pass

            logger.info(f"✓ Mixed precision enabled: {self.get_optimal_dtype()}")
            return autocast_context, scaler, cleanup

        except ImportError:
            logger.warning("Mixed precision not available")
            return None, None, lambda: None

    def create_optimizer(self, model, lr: float = 1e-3, weight_decay: float = 1e-4):
        """Create optimized optimizer"""
        # Use fused Adam if available (faster on GPU)
        try:
            import torch.optim as optim
            if self.config.available:
                return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
            else:
                return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        except:
            # Fallback to regular AdamW
            import torch.optim as optim
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def benchmark_gpu(self, model: torch.nn.Module, input_shape: tuple) -> Dict[str, float]:
        """Benchmark GPU performance"""
        if not self.config.available:
            return {'inference_ms': float('inf'), 'throughput': 0.0}

        model.eval()
        device = self.get_device()
        model = model.to(device)

        # Warm up
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        import time
        start_time = time.time()
        num_runs = 100

        torch.cuda.synchronize()  # Wait for all operations to complete
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        avg_time_ms = (elapsed / num_runs) * 1000
        throughput = num_runs / elapsed

        return {
            'inference_ms': avg_time_ms,
            'throughput': throughput,
            'device': str(device),
            'input_shape': input_shape
        }


# Global instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def get_device() -> torch.device:
    """Convenience function to get device"""
    return get_gpu_manager().get_device()


def gpu_available() -> bool:
    """Check if GPU is available"""
    return get_gpu_manager().config.available


def move_to_device(data: Any) -> Any:
    """Convenience function to move data to device"""
    return get_gpu_manager().move_to_device(data)


def empty_gpu_cache():
    """Convenience function to clear GPU cache"""
    get_gpu_manager().empty_cache()


# RTX 5070 Ti specific optimizations
RTX_5070_OPTIMIZATIONS = {
    'memory_efficient_attention': True,
    'flash_attention': True,  # Ada Lovelace supports Flash Attention
    'gradient_checkpointing': True,
    'fused_operations': True,
    'async_data_loading': True,
    'optimal_batch_sizes': {
        'lstm': 512,
        'transformer': 128,
        'cnn': 256,
        'mlp': 1024
    }
}




