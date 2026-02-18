#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cstdlib>

__global__ void axpy(float *x, float *y, const float a, int n) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n; i += stride) {
    y[i] = a * x[i] + y[i];
  }
}

float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
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

int main(void) {

  cudaEvent_t start_timer;
  cudaEvent_t stop_timer;

  float elapsed_time;

  int n = 1 << 28;
  size_t size = n * sizeof(float);

  float *host_x, *host_y, *host_z, *host_w;
  float *dev_x, *dev_y, *dev_z, *dev_w;

  host_x = (float *)malloc(size);
  host_y = (float *)malloc(size);
  host_z = (float *)malloc(size);
  host_w = (float *)malloc(size);

  // Allocate GPU memory for buffers
  cudaMalloc((void **)&dev_x, size);
  cudaMalloc((void **)&dev_y, size);
  cudaMalloc((void **)&dev_z, size);
  cudaMalloc((void **)&dev_w, size);

  // Generate input set
  srand(5347);

  for (int i = 0; i < n; i++) {
    host_x[i] = RandFloat(0.0, 1.0);
    host_y[i] = RandFloat(0.0, 1.0);
    host_z[i] = RandFloat(0.0, 1.0);
    host_w[i] = RandFloat(0.0, 1.0);
  }

  // Copy options data to GPU memory for further processing
  cudaMemcpy(dev_x, host_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, host_y, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_z, host_z, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_w, host_w, size, cudaMemcpyHostToDevice);

  int TPB = 256;
  int BPG = (n + TPB - 1) / TPB;

  // if (BPG > 65535)
  //   BPG = 65535;

  cudaEventCreate(&start_timer);
  cudaEventCreate(&stop_timer);
  cudaEventRecord(start_timer, 0);

  // axpy<<<BPG, TPB>>>(dev_x, dev_y, 1.1, n);
  // axpy<<<BPG, TPB>>>(dev_y, dev_z, 0.3, n);
  // axpy<<<BPG, TPB>>>(dev_z, dev_w, 0.9, n);
  fused_axpy<<<BPG, TPB>>>(dev_x, dev_y, dev_z, dev_w, 1.1, 0.3, 0.9, n);

  cudaEventRecord(stop_timer, 0);
  cudaEventSynchronize(stop_timer);
  cudaEventElapsedTime(&elapsed_time, start_timer, stop_timer);

  cudaMemcpy(host_w, dev_w, size, cudaMemcpyDeviceToHost);

  printf("peaking first 10 values\n");
  for (int i = 0; i < 10; i++) {
    printf("h_w[%d] = %f\n", i, host_w[i]);
  }
  printf("CUDA elapsed execution %f/ms\n", elapsed_time);

  free(host_x);
  free(host_y);
  free(host_z);
  free(host_w);
  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_z);
  cudaFree(dev_w);
  return 0;
}
