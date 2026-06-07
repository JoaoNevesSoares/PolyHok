
__device__
float anonymous_f020kk4kk0(float x, float _polyhok_ignored)
{
return ((x * x));
}

__device__ void* anonymous_f020kk4kk0_ptr = (void*) anonymous_f020kk4kk0;

extern "C" void* get_anonymous_f020kk4kk0_ptr()
{
	void* host_function_ptr;
	cudaMemcpyFromSymbol(&host_function_ptr, anonymous_f020kk4kk0_ptr, sizeof(void*));
	return host_function_ptr;
}


