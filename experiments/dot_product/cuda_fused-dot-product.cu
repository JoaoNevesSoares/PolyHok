#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

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
float anon_15e3ohc63n(float *x, float y, float z)
{
return (cas_float(x, y, z));
}


__device__
float anon_c2n03hh2n4(float arg0, float arg1)
{
return ((arg0 * arg1));
}


__device__
float anon_d221895ib8(float x, float y)
{
return ((x + y));
}


extern "C" __global__ void map2Reduce_kernel(float *t1, float *t2, float *ref_out, float initial, int n)
{
__shared__ float cache[256];
	int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
	int cacheIndex = threadIdx.x;
	int stride = (blockDim.x * gridDim.x);
	float temp = initial;
while((tid < n)){
	float mapped = anon_c2n03hh2n4(t1[tid], t2[tid]);
	temp = anon_d221895ib8(mapped, temp);
	tid = (tid + stride);
}
	cache[cacheIndex] = temp;
__syncthreads();
	int i = (blockDim.x / 2);
while((i != 0)){
if((cacheIndex < i))
{
	cache[cacheIndex] = anon_d221895ib8(cache[(cacheIndex + i)], cache[cacheIndex]);
}

__syncthreads();
	i = (i / 2);
}
if((cacheIndex == 0))
{
	float current_value = ref_out[0];
while((! (current_value == anon_15e3ohc63n(ref_out, current_value, anon_d221895ib8(cache[0], current_value))))){
	current_value = ref_out[0];
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
  map2Reduce_kernel<<<8192, 256>>>(dev_x,dev_y,res_red,0.0,n);
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
