
__device__
float anonymous_9be0knf4j9(float x, float _polyhok_ignored)
{
return ((x * x));
}

__device__ void* anonymous_9be0knf4j9_ptr = (void*) anonymous_9be0knf4j9;

extern "C" void* get_anonymous_9be0knf4j9_ptr()
{
	void* host_function_ptr;
	cudaMemcpyFromSymbol(&host_function_ptr, anonymous_9be0knf4j9_ptr, sizeof(void*));
	return host_function_ptr;
}


