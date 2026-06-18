#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <curand.h>

#define CUDA_CHECK(call)                                                \
do {                                                                    \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr,                                                   \
                "CUDA error in %s at %s:%d: %s\n",                      \
                #call, __FILE__, __LINE__,                              \
                cudaGetErrorString(err));                               \
        exit(EXIT_FAILURE);                                              \
    }                                                                   \
} while (0)

__global__ void scale(const float a, const float b, float *xs, int n) { 
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  int str = blockDim.x * gridDim.x; 
  for(int i = tid; i < n; i += str) { 
    xs[i] = a + (b - a) * xs[i];
  }
}

void gen_data(int n, float low, float high, float *dev_array) {
  curandGenerator_t gen;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateUniform(gen, (float*) dev_array, n);
  scale<<<(n + 127)/128, 128>>>((float) low,(float) high, (float *) dev_array, n);
  cudaDeviceSynchronize();
  curandDestroyGenerator(gen);
}

__global__ void axpy(float *x, float *y, const float a, int n) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride) {
    y[i] = a * x[i] + y[i];
  }
}

__global__ void fused_axpy(float *x, float *y, float *z, float *w, const float a, const float b, const float c, int n) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  float v = 0.0f;
  for (int i = tid; i < n; i += stride) {

    v = a * x[i] + y[i];
    v = b * v + z[i];
    w[i] = c * v + w[i];
  }
}

int main(int argc, char* argv[]) {

  cudaEvent_t start_timer;
  cudaEvent_t stop_timer;

  float elapsed_time;

  int n = 1 << 28;
  size_t size = n * sizeof(float);

  float *host_w;
  float *dev_x, *dev_y, *dev_z, *dev_w;

  host_w = (float *)malloc(size);

  // Allocate GPU memory for buffers
  CUDA_CHECK(cudaMalloc((void **)&dev_x, size));
  CUDA_CHECK(cudaMalloc((void **)&dev_y, size));
  CUDA_CHECK(cudaMalloc((void **)&dev_z, size));
  CUDA_CHECK(cudaMalloc((void **)&dev_w, size));

  // Generate input set
  gen_data(n, 0.0, 10.0,dev_x);
  gen_data(n, 0.0, 10.0,dev_y);
  gen_data(n, 0.0, 10.0,dev_z);

  int TPB = 256;
  int GRID = (n + TPB - 1) / TPB;

  CUDA_CHECK(cudaEventCreate(&start_timer));
  CUDA_CHECK(cudaEventCreate(&stop_timer));
  CUDA_CHECK(cudaEventRecord(start_timer, 0));

  axpy<<<GRID, TPB>>>(dev_x, dev_y, 1.1, n);
  axpy<<<GRID, TPB>>>(dev_y, dev_z, 0.3, n);
  axpy<<<GRID, TPB>>>(dev_z, dev_w, 0.9, n);
  // fused_axpy<<<BPG, TPB>>>(dev_x, dev_y, dev_z, dev_w, 1.1, 0.3, 0.9, n);

  CUDA_CHECK(cudaEventRecord(stop_timer, 0));
  CUDA_CHECK(cudaEventSynchronize(stop_timer));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_timer, stop_timer));

  CUDA_CHECK(cudaMemcpy(host_w, dev_w, size, cudaMemcpyDeviceToHost));

  printf("peaking first 10 values\n");
  for (int i = 0; i < 10; i++) {
    printf("h_w[%d] = %f\n", i, host_w[i]);
  }
  printf("CUDA elapsed execution %f/ms\n", elapsed_time);

  free(host_w);
  CUDA_CHECK(cudaFree(dev_x));
  CUDA_CHECK(cudaFree(dev_y));
  CUDA_CHECK(cudaFree(dev_z));
  CUDA_CHECK(cudaFree(dev_w));
  return 0;
}
