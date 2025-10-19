# GPU Benchmark Report
Generated: 20251019_143135

## Environment
- CUDA: ✅
- CuPy: ✅
- JAX: ❌
- TensorRT: ✅
- GPU: N/A

## JAX Benchmarks
- JAX not available

## TensorRT Micro-benchmark
- TensorRT not available: deserialize_cuda_engine(): incompatible function arguments. The following argument types are supported:
    1. (self: tensorrt_bindings.tensorrt.Runtime, serialized_engine: buffer) -> tensorrt_bindings.tensorrt.ICudaEngine
    2. (self: tensorrt_bindings.tensorrt.Runtime, stream_reader: tensorrt_bindings.tensorrt.IStreamReader) -> tensorrt_bindings.tensorrt.ICudaEngine
    3. (self: tensorrt_bindings.tensorrt.Runtime, stream_reader_v2: tensorrt_bindings.tensorrt.IStreamReaderV2) -> tensorrt_bindings.tensorrt.ICudaEngine

Invoked with: <tensorrt_bindings.tensorrt.Runtime object at 0x0000025C67A02EB0>, None

## VPIN Benchmark
- CPU: 332±15 ms
- GPU VPIN not available
