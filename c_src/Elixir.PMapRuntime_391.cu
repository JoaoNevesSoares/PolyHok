
__device__
float anonymous_8eoc3l0m9g(float x, float _polyhok_ignored)
{
return ((x * x));
}

__device__ void* anonymous_8eoc3l0m9g_ptr = (void*) anonymous_8eoc3l0m9g;

extern "C" void* get_anonymous_8eoc3l0m9g_ptr()
{
	void* host_function_ptr;
	cudaMemcpyFromSymbol(&host_function_ptr, anonymous_8eoc3l0m9g_ptr, sizeof(void*));
	return host_function_ptr;
}


