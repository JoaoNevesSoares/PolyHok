// Build:
//   nvcc -O3 -std=c++17 numba_fusion_cuda.cu -o numba_fusion_cuda
// Run:
//   ./numba_fusion_cuda

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int TPB = 256;
constexpr int N = 1 << 22;
constexpr int WARMUP = 1;
constexpr int ITERS = 10;

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(_err));                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int grid_1d(int n, int tpb = TPB) { return (n + tpb - 1) / tpb; }

__global__ void map_affine(const float* x, float* y, float a, float b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + b;
  }
}

__global__ void map_square(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = v * v;
  }
}

__global__ void map_fast_tanh_like(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    float av = fabsf(v);
    y[i] = v / (1.0f + av);
  }
}

__global__ void map_relu(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = (v > 0.0f) ? v : 0.0f;
  }
}

__global__ void fused_map_chain(const float* x, float* y, float a, float b,
                                float c, float d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = a * x[i] + b;
    v = v * v;
    float av = fabsf(v);
    v = v / (1.0f + av);
    v = c * v + d;
    y[i] = (v > 0.0f) ? v : 0.0f;
  }
}

template <typename LaunchFn>
float time_gpu(LaunchFn&& fn, int warmup = WARMUP, int iters = ITERS) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(iters);
}

struct BenchResult {
  float unfused_ms;
  float fused_ms;
};

BenchResult bench1_map_chain(const float* x_d, int n, float a = 1.1f,
                             float b = 0.3f, float c = 0.9f, float d = -0.2f) {
  float* y_d = nullptr;
  float* t1_d = nullptr;
  float* t2_d = nullptr;
  float* t3_d = nullptr;
  float* t4_d = nullptr;

  CHECK_CUDA(cudaMalloc(&y_d, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&t1_d, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&t2_d, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&t3_d, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&t4_d, static_cast<size_t>(n) * sizeof(float)));

  const int blocks = grid_1d(n);

  auto unfused = [&]() {
    map_affine<<<blocks, TPB>>>(x_d, t1_d, a, b, n);
    map_square<<<blocks, TPB>>>(t1_d, t2_d, n);
    map_fast_tanh_like<<<blocks, TPB>>>(t2_d, t3_d, n);
    map_affine<<<blocks, TPB>>>(t3_d, t4_d, c, d, n);
    map_relu<<<blocks, TPB>>>(t4_d, y_d, n);
    CHECK_CUDA(cudaGetLastError());
  };

  auto fused = [&]() {
    fused_map_chain<<<blocks, TPB>>>(x_d, y_d, a, b, c, d, n);
    CHECK_CUDA(cudaGetLastError());
  };

  float unfused_ms = time_gpu(unfused);
  float fused_ms = time_gpu(fused);

  CHECK_CUDA(cudaFree(t4_d));
  CHECK_CUDA(cudaFree(t3_d));
  CHECK_CUDA(cudaFree(t2_d));
  CHECK_CUDA(cudaFree(t1_d));
  CHECK_CUDA(cudaFree(y_d));

  return {unfused_ms, fused_ms};
}

std::vector<float> make_synthetic(int n, unsigned int seed = 0U) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> x(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = dist(rng);
  }
  return x;
}

}  // namespace

int main() {
  int device_count = 0;
  cudaError_t count_err = cudaGetDeviceCount(&device_count);
  if (count_err != cudaSuccess || device_count <= 0) {
    std::fprintf(
        stderr,
        "CUDA is not available. Check your driver, CUDA toolkit, and setup.\n");
    return EXIT_FAILURE;
  }

  std::printf("N = %d, dtype=float32, TPB = %d\n", N, TPB);

  std::vector<float> x = make_synthetic(N, 0U);

  float* x_d = nullptr;
  CHECK_CUDA(cudaMalloc(&x_d, static_cast<size_t>(N) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(x_d, x.data(), static_cast<size_t>(N) * sizeof(float),
                        cudaMemcpyHostToDevice));

  BenchResult result = bench1_map_chain(x_d, N);
  std::printf("[1] map chain, unfused %.3f ms, fused %.3f ms, speedup %.2fx\n",
              result.unfused_ms, result.fused_ms,
              result.unfused_ms / result.fused_ms);

  CHECK_CUDA(cudaFree(x_d));
  return EXIT_SUCCESS;
}
