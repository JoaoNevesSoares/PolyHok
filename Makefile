all: priv/gpu_nifs.so 

priv/gpu_nifs.so: c_src/gpu_nifs.cu
	nvcc --shared -g -lcuda -lnvrtc --compiler-options '-fPIC' -o priv/gpu_nifs.so c_src/gpu_nifs.cu

alternative: c_src/gpu_nifs.cu
	nvcc --shared -g \
  -cudart=none \
  -I/home/jans/.asdf/installs/erlang/26.2.1/usr/include \
  -L/usr/lib64 \
  -lcuda -lnvrtc -lcudart \
  --compiler-options '-fPIC' \
  -o priv/gpu_nifs.so c_src/gpu_nifs.cu

clean:
	rm priv/gpu_nifs.so
