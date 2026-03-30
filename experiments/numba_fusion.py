import math
import time
import numpy as np
from numba import cuda

TPB = 256

# -------------------------
# Map and zipWith kernels
# -------------------------

@cuda.jit
def map_affine(x, y, a, b, n):
    i = cuda.grid(1)
    if i < n:
        y[i] = a * x[i] + b

@cuda.jit
def map_square(x, y, n):
    i = cuda.grid(1)
    if i < n:
        v = x[i]
        y[i] = v * v

@cuda.jit
def map_fast_tanh_like(x, y, n):
    # cheap bounded nonlinearity: v / (1 + abs(v))
    i = cuda.grid(1)
    if i < n:
        v = x[i]
        av = abs(v)
        y[i] = v / (1.0 + av)

@cuda.jit
def map_relu(x, y, n):
    i = cuda.grid(1)
    if i < n:
        v = x[i]
        y[i] = v if v > 0.0 else 0.0


# -------------------------
# Fused elementwise kernels
# -------------------------

@cuda.jit
def fused_map_chain(x, y, a, b, c, d, n):
    # Example chain:
    #   v = a*x + b
    #   v = v*v
    #   v = v / (1 + abs(v))
    #   v = c*v + d
    #   v = relu(v)
    i = cuda.grid(1)
    if i < n:
        v = a * x[i] + b
        v = v * v
        av = abs(v)
        v = v / (1.0 + av)
        v = c * v + d
        y[i] = v if v > 0.0 else 0.0



# -------------------------
# Helpers
# -------------------------

def _grid_1d(n, tpb=TPB):
    return (n + tpb - 1) // tpb

def _make_scratch(n, dtype=np.float32):
    sizes = []
    cur = n
    while cur > 1:
        blocks = _grid_1d(cur)
        sizes.append(blocks)
        cur = blocks
    return [cuda.device_array(sz, dtype=dtype) for sz in sizes]

def time_gpu(fn, warmup=1, iters=10):
    # Measure elapsed time in milliseconds for fn(), using CUDA events.
    for _ in range(warmup):
        fn()
    cuda.synchronize()
    start = cuda.event()
    stop = cuda.event()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    stop.synchronize()
    return cuda.event_elapsed_time(start, stop) / iters

# -------------------------
# Benchmarks
# -------------------------

def bench1_map_chain(x_d, n, a=1.1, b=0.3, c=0.9, d=-0.2):
    y_d = cuda.device_array(n, dtype=np.float32)
    t1_d = cuda.device_array(n, dtype=np.float32)
    t2_d = cuda.device_array(n, dtype=np.float32)
    t3_d = cuda.device_array(n, dtype=np.float32)
    t4_d = cuda.device_array(n, dtype=np.float32)

    blocks = _grid_1d(n)

    def unfused():
        map_affine[blocks, TPB](x_d, t1_d, a, b, n)
        map_square[blocks, TPB](t1_d, t2_d, n)
        map_fast_tanh_like[blocks, TPB](t2_d, t3_d, n)
        map_affine[blocks, TPB](t3_d, t4_d, c, d, n)
        map_relu[blocks, TPB](t4_d, y_d, n)

    def fused():
        fused_map_chain[blocks, TPB](x_d, y_d, a, b, c, d, n)

    unfused_ms = time_gpu(unfused)
    fused_ms = time_gpu(fused)
    return unfused_ms, fused_ms

# -------------------------
# Synthetic input + runner
# -------------------------

def make_synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float32)
    y = rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float32)

    A = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    B = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    C = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    D = rng.normal(0.0, 1.0, size=n).astype(np.float32)

    return x, y, A, B, C, D

def main():
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your driver, CUDA toolkit, and Numba installation.")

    n = 2 ** 22  # change to 2**20, 2**24, etc, depending on your GPU memory
    print(f"N = {n}, dtype=float32, TPB = {TPB}")

    x, y, A, B, C, D = make_synthetic(n, seed=0)

    x_d = cuda.to_device(x)
    y_d = cuda.to_device(y)
    A_d = cuda.to_device(A)
    B_d = cuda.to_device(B)
    C_d = cuda.to_device(C)
    D_d = cuda.to_device(D)

    u1, f1 = bench1_map_chain(x_d, n)
    print(f"[1] map chain, unfused {u1:.3f} ms, fused {f1:.3f} ms, speedup {(u1 / f1):.2f}x")

if __name__ == "__main__":
    main()
