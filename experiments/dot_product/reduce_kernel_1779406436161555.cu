

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
float anon_g9124aol9i(float *x, float y, float z)
{
return (cas_float(x, y, z));
}


__device__
float sum(float x, float y)
{
return ((x + y));
}


extern "C" __global__ void reduce_kernel(float *a, float *ref4, float initial, int n)
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

