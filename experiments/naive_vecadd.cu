#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)

__device__ float sum(float x, float y) { return x + y; }

__global__ void vector_addition(float *A, float *B, float *C, int n) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    C[idx] = sum(A[idx], B[idx]);
  }
}

float cpu_serial(float *A, float *B, float *C, int n, int niter) {

  for(int iter = 0; iter < niter; iter++) {

    for(int i = 0; i < n; i++) {
      C[i] = A[i] + B[i];
    }
    memcpy(A,C,n * sizeof(float));
  }
  float sum = 0.0f;
  for(int i = 0; i < n; i++) {
    sum += C[i];
  }
  return sum;
}

int main(int argc, char **argv) {

  float *A, *B, *C;
  float *dA, *dB, *dC;

  int n = 1 << 5; 
  int nitter = 100;
  size_t size = n * sizeof(float);

  // Allocating CPU memory for options
  A = (float *)malloc(size);
  B = (float *)malloc(size);
  C = (float *)malloc(size);

  printf("teste debugging!\n");
  CUDA_CHECK(cudaMalloc((void **)&dA, size));
  CUDA_CHECK(cudaMalloc((void **)&dB, size));
  CUDA_CHECK(cudaMalloc((void **)&dC, size));

  // Generate options set
  for (int i = 0; i < n; i++) {
    A[i] = (float)i;
    B[i] = (float)n - i;
  }

  CUDA_CHECK(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice));

  int numThreads = 256;
  int numBlocks = (n + numThreads - 1) / numThreads;

  for(int iter = 0; iter < nitter; iter++) {
    vector_addition<<<numBlocks, numThreads>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dA, dC, size, cudaMemcpyDeviceToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost));

  printf("Checking the results...\n");
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += C[i];
  }
  printf("[GPU] sum of total add: %2f\n", sum);
  float sum_cpu = cpu_serial(A,B,C,n,nitter);
  printf("[CPU] sum of total add: %2f\n", sum_cpu);

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  free(A);
  free(B);
  free(C);
  return 0;
}
