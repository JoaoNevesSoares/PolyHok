

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


extern "C" __global__ void map2_kernel(float *a1, float *a2, float *a3, int size)
{
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
if((id < size))
{
	a3[id] = mult(a1[id], a2[id]);
}

}

