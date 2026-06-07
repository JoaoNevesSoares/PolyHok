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
float fusion_ce7f39c8c09d461e(float arg0)
{
	float __fuse_in_1 = (arg0 * 3.0);
	float __fuse_in_2 = (__fuse_in_1 + 1.0);
	float __fuse_in_3 = (__fuse_in_2 * 1.6);
	float __fuse_in_4 = (__fuse_in_3 - 0.25);
	float __fuse_in_5 = (__fuse_in_4 * __fuse_in_4);
	float __fuse_in_6 = (__fuse_in_5 + 0.43);
	float __fuse_in_7 = sqrtf(__fuse_in_6);
	float __fuse_in_8 = (__fuse_in_7 + 1.0);
	float __fuse_in_9 = (__fuse_in_8 / 0.43);
	float __fuse_in_10 = (__fuse_in_9 * 3.0);
	float __fuse_in_11 = (__fuse_in_10 - 0.25);
	float __fuse_in_12 = (__fuse_in_11 * __fuse_in_11);
	float __fuse_in_13 = (__fuse_in_12 + 0.43);
	float __fuse_in_14 = sqrtf(__fuse_in_13);
	float __fuse_in_15 = (__fuse_in_14 - 0.25);
	float __fuse_in_16 = (__fuse_in_15 + 1.0);
	float __fuse_in_17 = (__fuse_in_16 / 0.43);
	float __fuse_in_18 = (__fuse_in_17 * 1.6);
	float __fuse_in_19 = (__fuse_in_18 + 0.43);
return ((__fuse_in_19 / 0.43));
}


extern "C" __global__ void map_ker(float *a1, float *a2, int size)
{
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int stride = (blockDim.x * gridDim.x);
for( int i = index; i<size; i+=stride){
	a2[i] = fusion_ce7f39c8c09d461e(a1[i]);
}

}

