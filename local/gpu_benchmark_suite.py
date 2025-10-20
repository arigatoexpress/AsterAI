#!/usr/bin/env python3
"""
Unified GPU Benchmark Suite: JAX, TensorRT, and VPIN

Capabilities:
- Detect GPU availability and environment
- JAX benchmarks (JIT ops, reductions, matmul) with CPU vs GPU comparison
- TensorRT micro-benchmark (engine build + inference) if available
- VPIN (Volume-synchronized PIN) computation on CPU (NumPy) vs GPU (CuPy) for speedup
- Structured JSON and Markdown reports

Notes:
- All GPU dependencies are optional. The suite gracefully degrades to CPU.
- Designed for RTX 5070 Ti Blackwell but works on other CUDA GPUs.
"""

import os
import sys
import json
import time
import math
import logging
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EnvStatus:
    cuda_available: bool
    cupy_available: bool
    jax_available: bool
    tensorrt_available: bool
    gpu_name: Optional[str]
    total_memory_gb: Optional[float]
    free_memory_gb: Optional[float]


def detect_environment() -> EnvStatus:
    cuda_available = False
    cupy_available = False
    jax_available = False
    tensorrt_available = False
    gpu_name = None
    total_memory_gb = None
    free_memory_gb = None

    try:
        import cupy as cp  # type: ignore
        cupy_available = True
        try:
            cuda_available = cp.cuda.is_available()
        except Exception:
            cuda_available = True  # If CuPy imports, we assume CUDA present

        try:
            dev = cp.cuda.Device()
            props = dev.attributes
            gpu_name = dev.name
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            total_memory_gb = total_b / (1024 ** 3)
            free_memory_gb = free_b / (1024 ** 3)
        except Exception:
            pass
    except Exception:
        pass

    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # noqa: F401
        jax_available = True
        # If JAX is available but defaulting to CPU, still fine; we time it.
        os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')
    except Exception:
        pass

    try:
        import tensorrt as trt  # type: ignore  # noqa: F401
        tensorrt_available = True
    except Exception:
        pass

    return EnvStatus(
        cuda_available=cuda_available,
        cupy_available=cupy_available,
        jax_available=jax_available,
        tensorrt_available=tensorrt_available,
        gpu_name=gpu_name,
        total_memory_gb=total_memory_gb,
        free_memory_gb=free_memory_gb,
    )


def run_trials_ms(fn, warmup: int, repeat: int) -> Tuple[float, float, List[float]]:
    for _ in range(max(0, warmup)):
        fn()
    samples: List[float] = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        samples.append((end - start) * 1000.0)
    mean = float(np.mean(samples)) if samples else float('nan')
    std = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    return mean, std, samples


def ci95_ms(std_ms: float, n: int) -> float:
    if n and n > 1:
        return 1.96 * (std_ms / math.sqrt(n))
    return 0.0


def jax_benchmarks(env: EnvStatus, warmup: int = 3, repeat: int = 10, shapes: Tuple[int, int, int] = (2048, 1024, 2048)) -> Dict[str, Any]:
    results: Dict[str, Any] = {"available": False}
    if not env.jax_available:
        return results

    import jax
    import jax.numpy as jnp

    results["available"] = True
    results["backend"] = jax.default_backend()

    # Shapes chosen to be meaningfully large but not excessive
    m, k, n = shapes

    # CPU arrays
    x_cpu = np.random.randn(m, k).astype(np.float32)
    y_cpu = np.random.randn(k, n).astype(np.float32)

    # JAX device arrays
    x = jnp.array(x_cpu)
    y = jnp.array(y_cpu)

    @jax.jit
    def matmul(a, b):
        return a @ b

    @jax.jit
    def reductions(a):
        return jnp.sum(a), jnp.mean(a), jnp.std(a)

    # Warm-up to trigger compilation
    _ = matmul(x, y).block_until_ready()
    _ = reductions(x).block_until_ready()

    # Measure JAX GPU/CPU time (depending on backend)
    jax_matmul_mean, jax_matmul_std, jax_matmul_samples = run_trials_ms(lambda: matmul(x, y).block_until_ready(), warmup, repeat)
    jax_reduce_mean, jax_reduce_std, jax_reduce_samples = run_trials_ms(lambda: reductions(x).block_until_ready(), warmup, repeat)

    # NumPy CPU baselines
    numpy_matmul_mean, numpy_matmul_std, numpy_matmul_samples = run_trials_ms(lambda: np.matmul(x_cpu, y_cpu), 0, max(1, repeat // 2))
    numpy_reduce_mean, numpy_reduce_std, numpy_reduce_samples = run_trials_ms(lambda: (np.sum(x_cpu), np.mean(x_cpu), np.std(x_cpu)), 0, max(1, repeat))

    results["matmul_ms"] = {
        "jax_mean": jax_matmul_mean,
        "jax_std": jax_matmul_std,
        "jax_samples": jax_matmul_samples,
        "jax_ci95": ci95_ms(jax_matmul_std, len(jax_matmul_samples)),
        "numpy_cpu_mean": numpy_matmul_mean,
        "numpy_cpu_std": numpy_matmul_std,
        "numpy_samples": numpy_matmul_samples,
        "numpy_ci95": ci95_ms(numpy_matmul_std, len(numpy_matmul_samples)),
        "speedup_x": (numpy_matmul_mean / jax_matmul_mean) if jax_matmul_mean and jax_matmul_mean > 0 else None,
        "shape": [m, k, n],
        "repeat": repeat,
        "warmup": warmup,
    }
    results["reductions_ms"] = {
        "jax_mean": jax_reduce_mean,
        "jax_std": jax_reduce_std,
        "jax_samples": jax_reduce_samples,
        "jax_ci95": ci95_ms(jax_reduce_std, len(jax_reduce_samples)),
        "numpy_cpu_mean": numpy_reduce_mean,
        "numpy_cpu_std": numpy_reduce_std,
        "numpy_samples": numpy_reduce_samples,
        "numpy_ci95": ci95_ms(numpy_reduce_std, len(numpy_reduce_samples)),
        "speedup_x": (numpy_reduce_mean / jax_reduce_mean) if jax_reduce_mean and jax_reduce_mean > 0 else None,
        "repeat": repeat,
        "warmup": warmup,
    }

    return results


def tensorrt_benchmark(env: EnvStatus, batch_size: int = 1024, warmup: int = 10, repeat: int = 50) -> Dict[str, Any]:
    results: Dict[str, Any] = {"available": False}
    if not env.tensorrt_available:
        return results

    try:
        import tensorrt as trt
        import numpy as np
        import ctypes

        # TensorRT requires a runtime + engine + context; we build a tiny network
        logger.info("Building TensorRT engine for micro-benchmark ...")
        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB

        # Define a simple network: y = ReLU(Wx + b)
        input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(-1, 1024))

        # Constants
        w = np.random.randn(1024, 1024).astype(np.float32)
        b = np.random.randn(1024).astype(np.float32)
        weight = network.add_constant(w.shape, w)
        bias = network.add_constant(b.shape, b)

        matmul = network.add_matrix_multiply(input_tensor, trt.MatrixOperation.NONE, weight.get_output(0), trt.MatrixOperation.NONE)
        bias_add = network.add_elementwise(matmul.get_output(0), bias.get_output(0), trt.ElementWiseOperation.SUM)
        relu = network.add_activation(bias_add.get_output(0), trt.ActivationType.RELU)
        relu.get_output(0).name = "output"
        network.mark_output(relu.get_output(0))

        serialized = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(logger_trt)
        engine = runtime.deserialize_cuda_engine(serialized)
        context = engine.create_execution_context()

        # Allocate buffers
        input_shape = (batch_size, 1024)
        output_shape = (batch_size, 1024)

        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401

        h_in = np.random.randn(*input_shape).astype(np.float32)
        h_out = np.empty(output_shape, dtype=np.float32)
        d_in = cuda.mem_alloc(h_in.nbytes)
        d_out = cuda.mem_alloc(h_out.nbytes)

        bindings = [int(d_in), int(d_out)]

        def infer_once():
            cuda.memcpy_htod(d_in, h_in)
            context.execute_v2(bindings)
            cuda.memcpy_dtoh(h_out, d_out)

        # Warm-up + trials
        mean_ms, std_ms, samples = run_trials_ms(infer_once, warmup, repeat)
        results["available"] = True
        results["inference_ms_per_batch_mean"] = mean_ms
        results["inference_ms_per_batch_std"] = std_ms
        results["inference_ms_per_batch_ci95"] = ci95_ms(std_ms, len(samples))
        results["samples_ms"] = samples
        results["throughput_samples_per_s"] = (batch_size / (mean_ms / 1000.0)) if mean_ms and mean_ms > 0 else None
        results["batch_size"] = batch_size

        return results
    except Exception as e:
        results["error"] = str(e)
        return results


def vpin_compute_cpu(prices: np.ndarray, volumes: np.ndarray, bucket_volume: float = 1e5) -> float:
    # VPIN per Easley, Lopez de Prado, O'Hara: bucket trades by equal volume, classify buys/sells by price change sign.
    # This is a simplified approximation.
    assert prices.ndim == 1 and volumes.ndim == 1 and prices.size == volumes.size

    # Build buckets
    v_cum = 0.0
    buy_vol = 0.0
    sell_vol = 0.0
    total_imbalance = 0.0
    bucket_count = 0

    prev_price = prices[0]
    for i in range(1, prices.size):
        v = float(volumes[i])
        direction = 1.0 if prices[i] >= prev_price else -1.0
        buy_vol += v * (1.0 if direction > 0 else 0.0)
        sell_vol += v * (1.0 if direction < 0 else 0.0)
        v_cum += v
        prev_price = prices[i]
        if v_cum >= bucket_volume:
            total_imbalance += abs(buy_vol - sell_vol)
            bucket_count += 1
            v_cum = 0.0
            buy_vol = 0.0
            sell_vol = 0.0

    if bucket_count == 0:
        return 0.0
    return total_imbalance / (bucket_count * bucket_volume)


def vpin_benchmarks(env: EnvStatus, bucket_volume: float = 5e5, warmup: int = 0, repeat: int = 3) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    n = 2_000_000
    rng = np.random.default_rng(42)
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0, 0.001, n)).astype(np.float64)
    volumes = rng.uniform(1.0, 500.0, n).astype(np.float64)

    cpu_mean, cpu_std, cpu_samples = run_trials_ms(lambda: vpin_compute_cpu(prices, volumes, bucket_volume), warmup, repeat)
    results["cpu_ms_mean"] = cpu_mean
    results["cpu_ms_std"] = cpu_std
    results["cpu_samples_ms"] = cpu_samples
    results["cpu_ms_ci95"] = ci95_ms(cpu_std, len(cpu_samples))

    # GPU via CuPy vectorized approximation
    if env.cupy_available and env.cuda_available:
        try:
            import cupy as cp

            prices_gpu = cp.asarray(prices)
            volumes_gpu = cp.asarray(volumes)

            def vpin_gpu_once():
                # Compute returns sign
                price_diff = cp.diff(prices_gpu, prepend=prices_gpu[0])
                direction = cp.sign(price_diff)
                buy_vol = cp.where(direction >= 0, volumes_gpu, 0.0)
                sell_vol = cp.where(direction < 0, volumes_gpu, 0.0)

                # Inclusive scan of volumes for bucket boundaries
                v_cum = cp.cumsum(volumes_gpu)
                bucket_ids = cp.floor(v_cum / 5e5)

                # Group by bucket id and sum
                max_bucket = int(cp.asnumpy(bucket_ids[-1]))
                if max_bucket == 0:
                    return 0.0
                # Use scatter-add via bincount
                buy_sum = cp.bincount(bucket_ids.astype(cp.int32), weights=buy_vol, minlength=max_bucket + 1)
                sell_sum = cp.bincount(bucket_ids.astype(cp.int32), weights=sell_vol, minlength=max_bucket + 1)
                imbalance = cp.abs(buy_sum - sell_sum)
                return cp.asnumpy(cp.sum(imbalance) / ((max_bucket + 1) * 5e5))

            # Warm-up + trials
            gpu_mean, gpu_std, gpu_samples = run_trials_ms(vpin_gpu_once, max(1, warmup), repeat)
            results["gpu_ms_mean"] = gpu_mean
            results["gpu_ms_std"] = gpu_std
            results["gpu_samples_ms"] = gpu_samples
            results["gpu_ms_ci95"] = ci95_ms(gpu_std, len(gpu_samples))
            results["speedup_x"] = (cpu_mean / gpu_mean) if gpu_mean and gpu_mean > 0 else None
        except Exception as e:
            results["gpu_error"] = str(e)
    else:
        results["gpu_ms_mean"] = None
        results["speedup_x"] = None

    return results


def generate_reports(env: EnvStatus, jax_res: Dict[str, Any], trt_res: Dict[str, Any], vpin_res: Dict[str, Any]) -> Tuple[str, str]:
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    json_path = f"gpu_benchmark_report_{timestamp}.json"
    md_path = f"gpu_benchmark_report_{timestamp}.md"

    report = {
        "environment": {
            "cuda_available": env.cuda_available,
            "cupy_available": env.cupy_available,
            "jax_available": env.jax_available,
            "tensorrt_available": env.tensorrt_available,
            "gpu_name": env.gpu_name,
            "total_memory_gb": env.total_memory_gb,
            "free_memory_gb": env.free_memory_gb,
        },
        "jax": jax_res,
        "tensorrt": trt_res,
        "vpin": vpin_res,
        "generated_at": timestamp,
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    lines = []
    lines.append(f"# GPU Benchmark Report\n")
    lines.append(f"Generated: {timestamp}\n")
    lines.append("\n## Environment\n")
    lines.append(f"- CUDA: {'✅' if env.cuda_available else '❌'}\n")
    lines.append(f"- CuPy: {'✅' if env.cupy_available else '❌'}\n")
    lines.append(f"- JAX: {'✅' if env.jax_available else '❌'}\n")
    lines.append(f"- TensorRT: {'✅' if env.tensorrt_available else '❌'}\n")
    lines.append(f"- GPU: {env.gpu_name or 'N/A'}\n")
    if env.total_memory_gb is not None:
        lines.append(f"- Memory: {env.free_memory_gb:.1f}GB free / {env.total_memory_gb:.1f}GB total\n")

    lines.append("\n## JAX Benchmarks\n")
    if jax_res.get("available"):
        lines.append(f"- Backend: {jax_res.get('backend')}\n")
        mm = jax_res.get("matmul_ms", {})
        rd = jax_res.get("reductions_ms", {})
        if mm:
            speed = mm.get("speedup_x")
            lines.append(
                f"- Matmul (shape {tuple(mm.get('shape', []))}, n={mm.get('repeat', 0)}): "
                f"JAX {mm.get('jax_mean', float('nan')):.1f}±{mm.get('jax_std', 0.0):.1f} ms vs "
                f"NumPy {mm.get('numpy_cpu_mean', float('nan')):.1f}±{mm.get('numpy_cpu_std', 0.0):.1f} ms — "
                f"Speedup {speed:.2f}x\n"
            )
        if rd:
            lines.append(
                f"- Reductions (n={rd.get('repeat', 0)}): JAX {rd.get('jax_mean', float('nan')):.2f}±{rd.get('jax_std', 0.0):.2f} ms vs "
                f"NumPy {rd.get('numpy_cpu_mean', float('nan')):.2f}±{rd.get('numpy_cpu_std', 0.0):.2f} ms — "
                f"Speedup {rd.get('speedup_x'):.2f}x\n"
            )
    else:
        lines.append("- JAX not available\n")

    lines.append("\n## TensorRT Micro-benchmark\n")
    if trt_res.get("available"):
        lines.append(
            f"- Inference per batch: {trt_res.get('inference_ms_per_batch_mean'):.2f}±{trt_res.get('inference_ms_per_batch_std', 0.0):.2f} ms "
            f"(batch={trt_res.get('batch_size')})\n"
        )
        tput = trt_res.get("throughput_samples_per_s")
        if tput is not None:
            lines.append(f"- Throughput: {tput:.0f} samples/s\n")
    else:
        lines.append(f"- TensorRT not available{': ' + trt_res.get('error') if trt_res.get('error') else ''}\n")

    lines.append("\n## VPIN Benchmark\n")
    lines.append(f"- CPU: {vpin_res.get('cpu_ms_mean', float('nan')):.0f}±{vpin_res.get('cpu_ms_std', 0.0):.0f} ms\n")
    if vpin_res.get("gpu_ms_mean") is not None:
        lines.append(f"- GPU: {vpin_res.get('gpu_ms_mean'):.0f}±{vpin_res.get('gpu_ms_std', 0.0):.0f} ms\n")
        if vpin_res.get("speedup_x"):
            lines.append(f"- Speedup: {vpin_res.get('speedup_x'):.2f}x\n")
    else:
        lines.append("- GPU VPIN not available\n")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("".join(lines))

    return json_path, md_path


def write_csv_logs(timestamp: str, env: EnvStatus, jax_res: Dict[str, Any], trt_res: Dict[str, Any], vpin_res: Dict[str, Any]) -> Optional[str]:
    try:
        import csv
        os.makedirs('benchmarks', exist_ok=True)
        path = os.path.join('benchmarks', f"gpu_benchmark_trials_{timestamp}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "suite", "metric", "detail", "trial_index", "value_ms"])
            # JAX
            if jax_res.get("available"):
                mm = jax_res.get("matmul_ms", {})
                for i, v in enumerate(mm.get("jax_samples", []) or []):
                    writer.writerow([timestamp, "jax", "matmul_jax", str(mm.get("shape")), i, v])
                for i, v in enumerate(mm.get("numpy_samples", []) or []):
                    writer.writerow([timestamp, "numpy", "matmul_cpu", str(mm.get("shape")), i, v])
                rd = jax_res.get("reductions_ms", {})
                for i, v in enumerate(rd.get("jax_samples", []) or []):
                    writer.writerow([timestamp, "jax", "reductions_jax", "sum-mean-std", i, v])
                for i, v in enumerate(rd.get("numpy_samples", []) or []):
                    writer.writerow([timestamp, "numpy", "reductions_cpu", "sum-mean-std", i, v])
            # TensorRT
            if trt_res.get("available"):
                for i, v in enumerate(trt_res.get("samples_ms", []) or []):
                    writer.writerow([timestamp, "tensorrt", "inference_batch", str(trt_res.get("batch_size")), i, v])
            # VPIN
            for i, v in enumerate(vpin_res.get("cpu_samples_ms", []) or []):
                writer.writerow([timestamp, "vpin", "cpu", "bucket=5e5", i, v])
            if vpin_res.get("gpu_samples_ms"):
                for i, v in enumerate(vpin_res.get("gpu_samples_ms", []) or []):
                    writer.writerow([timestamp, "vpin", "gpu", "bucket=5e5", i, v])
        return path
    except Exception:
        return None


def main() -> int:
    logger.info("Detecting environment ...")
    parser = argparse.ArgumentParser(description="Unified GPU Benchmark Suite")
    parser.add_argument('--jax-shape', type=str, default='2048,1024,2048', help='Matmul shape m,k,n for JAX')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations')
    parser.add_argument('--repeat', type=int, default=10, help='Repeat iterations')
    parser.add_argument('--trt-batch', type=int, default=1024, help='TensorRT batch size')
    parser.add_argument('--vpin-repeat', type=int, default=3, help='VPIN repeat iterations')
    args = parser.parse_args()

    try:
        m, k, n = [int(p) for p in args.jax_shape.split(',')]
    except Exception:
        m, k, n = 2048, 1024, 2048

    env = detect_environment()
    logger.info(
        "CUDA=%s, CuPy=%s, JAX=%s, TensorRT=%s, GPU=%s",
        env.cuda_available, env.cupy_available, env.jax_available, env.tensorrt_available, env.gpu_name,
    )

    logger.info("Running JAX benchmarks ...")
    jax_res = jax_benchmarks(env, warmup=args.warmup, repeat=args.repeat, shapes=(m, k, n))

    logger.info("Running TensorRT micro-benchmark ...")
    trt_res = tensorrt_benchmark(env, batch_size=args.trt_batch, warmup=args.warmup, repeat=args.repeat)

    logger.info("Running VPIN benchmarks ...")
    vpin_res = vpin_benchmarks(env, warmup=0, repeat=args.vpin_repeat)

    json_path, md_path = generate_reports(env, jax_res, trt_res, vpin_res)
    csv_path = write_csv_logs(time.strftime('%Y%m%d_%H%M%S'), env, jax_res, trt_res, vpin_res)
    logger.info("Reports generated: %s, %s", json_path, md_path)
    if csv_path:
        logger.info("CSV trials log: %s", csv_path)
    print("\n==== GPU BENCHMARK SUMMARY ====")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    if csv_path:
        print(f"CSV trials: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


