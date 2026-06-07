
__device__
float anonymous_90l9ji5go5(float x, float _polyhok_ignored)
{
return ((x * x));
}

__device__ void* anonymous_90l9ji5go5_ptr = (void*) anonymous_90l9ji5go5;

extern "C" void* get_anonymous_90l9ji5go5_ptr()
{
	void* host_function_ptr;
	cudaMemcpyFromSymbol(&host_function_ptr, anonymous_90l9ji5go5_ptr, sizeof(void*));
	return host_function_ptr;
}


