// Build:
//   nvcc -O3 -std=c++17 experiments/dp_fusion_cuda.cu -o experiments/dp_fusion_cuda
// Run:
//   ./experiments/dp_fusion_cuda

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int N = 4'194'304;
constexpr int BLOCK_SIZE = 256;
constexpr int GRID_SIZE = 4096;
constexpr int EXECUTIONS = 30;
constexpr float LOW = -1.0f;
constexpr float HIGH = 1.0f;
constexpr float TOLERANCE = 1.0e-6f;

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(_err));                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void mult_kernel(const float* input, float* temp, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride) {
    temp[i] = input[i] * 3.0f;
  }
}

__global__ void add_kernel(const float* temp, float* output, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride) {
    output[i] = temp[i] + 1.0f;
  }
}

__global__ void fused_mult_add_kernel(const float* input, float* output,
                                      int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride) {
    output[i] = input[i] * 3.0f + 1.0f;
  }
}

std::vector<float> random_tensor(int n, float low, float high,
                                 unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> vals(static_cast<size_t>(n));

  for (int i = 0; i < n; ++i) {
    float t = dist(rng);
    vals[static_cast<size_t>(i)] = (1.0f - t) * low + t * high;
  }

  return vals;
}

float time_unfused(const float* dev_input, float* dev_temp, float* dev_output,
                   int n) {
  cudaEvent_t start_timer;
  cudaEvent_t stop_timer;
  CHECK_CUDA(cudaEventCreate(&start_timer));
  CHECK_CUDA(cudaEventCreate(&stop_timer));

  CHECK_CUDA(cudaEventRecord(start_timer, 0));
  mult_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_input, dev_temp, n);
  CHECK_CUDA(cudaGetLastError());
  add_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_temp, dev_output, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(stop_timer, 0));
  CHECK_CUDA(cudaEventSynchronize(stop_timer));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_timer, stop_timer));
  CHECK_CUDA(cudaEventDestroy(start_timer));
  CHECK_CUDA(cudaEventDestroy(stop_timer));
  return elapsed_ms;
}

float time_fused(const float* dev_input, float* dev_output, int n) {
  cudaEvent_t start_timer;
  cudaEvent_t stop_timer;
  CHECK_CUDA(cudaEventCreate(&start_timer));
  CHECK_CUDA(cudaEventCreate(&stop_timer));

  CHECK_CUDA(cudaEventRecord(start_timer, 0));
  fused_mult_add_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_input, dev_output, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(stop_timer, 0));
  CHECK_CUDA(cudaEventSynchronize(stop_timer));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_timer, stop_timer));
  CHECK_CUDA(cudaEventDestroy(start_timer));
  CHECK_CUDA(cudaEventDestroy(stop_timer));
  return elapsed_ms;
}

float max_abs_error(const std::vector<float>& input,
                    const std::vector<float>& output) {
  float max_error = 0.0f;

  for (int i = 0; i < N; ++i) {
    float expected = input[static_cast<size_t>(i)] * 3.0f + 1.0f;
    float error = std::fabs(output[static_cast<size_t>(i)] - expected);
    if (error > max_error) {
      max_error = error;
    }
  }

  return max_error;
}

void check_error_or_exit(const char* label, float max_error) {
  std::printf("%s max abs error: %.9g\n", label, max_error);
  if (max_error > TOLERANCE) {
    std::fprintf(stderr, "%s validation failed: max abs error %.9g > %.9g\n",
                 label, max_error, TOLERANCE);
    std::exit(EXIT_FAILURE);
  }
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

  const size_t size = static_cast<size_t>(N) * sizeof(float);

  std::vector<float> host_x = random_tensor(N, LOW, HIGH, 5347U);
  std::vector<float> host_out(static_cast<size_t>(N));

  float* dev_x = nullptr;
  float* dev_tmp = nullptr;
  float* dev_out = nullptr;

  CHECK_CUDA(cudaMalloc(&dev_x, size));
  CHECK_CUDA(cudaMalloc(&dev_tmp, size));
  CHECK_CUDA(cudaMalloc(&dev_out, size));
  CHECK_CUDA(cudaMemcpy(dev_x, host_x.data(), size, cudaMemcpyHostToDevice));

  std::printf("n = %d, dtype = float32, grid = %d, block = %d\n", N,
              GRID_SIZE, BLOCK_SIZE);

  for (int execution = 1; execution <= EXECUTIONS; ++execution) {
    std::printf("Execucao: %d\n", execution);

    float unfused_ms = time_unfused(dev_x, dev_tmp, dev_out, N);
    CHECK_CUDA(cudaMemcpy(host_out.data(), dev_out, size,
                          cudaMemcpyDeviceToHost));
    float unfused_error = max_abs_error(host_x, host_out);

    std::printf("Run unfused\n");
    std::printf("CUDA\t%d\t%.3f ms\n", N, unfused_ms);
    check_error_or_exit("unfused", unfused_error);

    float fused_ms = time_fused(dev_x, dev_out, N);
    CHECK_CUDA(cudaMemcpy(host_out.data(), dev_out, size,
                          cudaMemcpyDeviceToHost));
    float fused_error = max_abs_error(host_x, host_out);

    std::printf("Run fused\n");
    std::printf("CUDA\t%d\t%.3f ms\n", N, fused_ms);
    check_error_or_exit("fused", fused_error);
  }

  CHECK_CUDA(cudaFree(dev_out));
  CHECK_CUDA(cudaFree(dev_tmp));
  CHECK_CUDA(cudaFree(dev_x));

  return EXIT_SUCCESS;
}
