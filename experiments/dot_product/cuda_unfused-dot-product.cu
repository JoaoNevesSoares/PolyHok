#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>



// MAP KERNEL
__device__ static float cas_float(float* address, float oldv, float newv)
{
    int* address_as_i = (int*) address;
    return  __int_as_float(atomicCAS(address_as_i, __float_as_int(oldv), __float_as_int(newv)));
}

__device__ static int cas_int(int* address, int oldv, int newv)
{
    return  atomicCAS(address, oldv, newv);
}

__device__ static double cas_double(double* address, double oldv, double newv)
{
    unsigned long long int * address_as_i = (unsigned long long int *) address;
    return  __longlong_as_double(  atomicCAS(address_as_i, __double_as_longlong(oldv),__double_as_longlong(newv))  );
}

__device__
float mult(float x, float y)
{
return ((x * y));
}


__global__ void map2_kernel(float *a1, float *a2, float *a3, int size)
{
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
if((id < size))
{
	a3[id] = mult(a1[id], a2[id]);
}

}


//REDUCE KERNEL

__device__
float anon_g9124aol9i(float *x, float y, float z)
{
return (cas_float(x, y, z));
}


__device__
float sum(float x, float y)
{
return ((x + y));
}


__global__ void reduce_kernel(float *a, float *ref4, float initial, int n)
{
__shared__ float cache[256];
	int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
	int cacheIndex = threadIdx.x;
	float temp = initial;
while((tid < n)){
	temp = sum(a[tid], temp);
	tid = ((blockDim.x * gridDim.x) + tid);
}
	cache[cacheIndex] = temp;
__syncthreads();
	int i = (blockDim.x / 2);
while((i != 0)){
if((cacheIndex < i))
{
	cache[cacheIndex] = sum(cache[(cacheIndex + i)], cache[cacheIndex]);
}

__syncthreads();
	i = (i / 2);
}
if((cacheIndex == 0))
{
	float current_value = ref4[0];
while((! (current_value == anon_g9124aol9i(ref4, current_value, sum(cache[0], current_value))))){
	current_value = ref4[0];
}
}

}


float random_float(float min_value, float max_value) {
  float normalized = (float)rand() / (float)RAND_MAX;
  return min_value + normalized * (max_value - min_value);
}

// code that generates random bodies ---
void initialize_host_data(float *x, int n, int random_seed) {
  srand(random_seed);
  for (int i = 0; i < n; i++) {
    x[i] = random_float(-1.0f, 1.0f);
  }
}

int compute_grid_size(int num_bodies, int block_size) {
  return (num_bodies + block_size - 1) / block_size;
}

int main(int argc, char **argv) {

  int n = 50000000;

  float *host_x = (float *)malloc(sizeof(float) * n);
  float *host_y = (float *)malloc(sizeof(float) * n);

  initialize_host_data(host_x, n, 1234);
  initialize_host_data(host_y, n, 1234);
  float *dev_x, *dev_y, *res_map, *res_red;

  cudaMalloc(&dev_x, sizeof(float) * n);
  cudaMalloc(&dev_y, sizeof(float) * n);
  cudaMalloc(&res_map, sizeof(float) * n);
  cudaMalloc(&res_red, sizeof(float) * n);

  cudaMemcpy(dev_x, host_x, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, host_y, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemset(res_map, 0, sizeof(float) * n);
  cudaMemset(res_red, 0, sizeof(float) * n);


  const int grid_size = compute_grid_size((int)n, 256);
  float milliseconds = 0.0f;

  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent, 0);

  // PUT HERE THE CODE FOR THE KERNEL
  //
  map2_kernel<<<grid_size, 256>>>(dev_x,dev_y, res_map, n);
  reduce_kernel<<<4096, 256>>>(res_map, res_red, 0.0, n);

  cudaMemcpy(host_x, res_red, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  printf("TOTAL TIME: %f\n ", milliseconds);
  printf("res = %f\n", host_x[0]);

  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(res_map);
  cudaFree(res_red);
  free(host_x);
  free(host_y);

  return 0;
}
