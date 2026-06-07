

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

