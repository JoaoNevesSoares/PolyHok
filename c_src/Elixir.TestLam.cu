
__device__
float anonymous_fb9ooee40e(float x, float y)
{
return ((x * y));
}

__device__ void* anonymous_fb9ooee40e_ptr = (void*) anonymous_fb9ooee40e;

extern "C" void* get_anonymous_fb9ooee40e_ptr()
{
	void* host_function_ptr;
	cudaMemcpyFromSymbol(&host_function_ptr, anonymous_fb9ooee40e_ptr, sizeof(void*));
	return host_function_ptr;
}


