all: priv/gpu_nifs.so 

priv/gpu_nifs.so: c_src/gpu_nifs.cu
	nvcc --shared -g -lcuda -lnvrtc --compiler-options '-fPIC' -o priv/gpu_nifs.so c_src/gpu_nifs.cu