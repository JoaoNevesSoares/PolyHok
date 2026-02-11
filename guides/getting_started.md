# Getting Started

This repo is an **Elixir-to-CUDA DSL/runtime** for writing GPU kernels in Elixir and executing them via NIFs.

## What it is

- Core API is in `lib/poly_hok.ex:1`.
- It loads native CUDA NIFs on module load (`@on_load`) and exposes:
  - GPU array lifecycle (`new_gnx`, `get_gnx`) in `lib/poly_hok.ex:170`, `lib/poly_hok.ex:255`
  - Kernel launch (`spawn`) in `lib/poly_hok.ex:703`
  - GPU lambda macros (`phok`, `clo`) in `lib/poly_hok.ex:31`, `lib/poly_hok.ex:7`
  - DSL module capture (`PolyHok.defmodule`) in `lib/poly_hok.ex:105`

## How the architecture fits together

- `lib/poly_hok/ske.ex:3` defines high-level skeletons (`map`, `map2`, `map3`, `reduce`, `mapReduce`, `map2Reduce`) and the underlying kernels (`defk ...`).
- `lib/fusion.ex:210` adds AST-level fusion (`with_fusion`) to combine map/reduce pipelines into fewer kernels.
- `lib/poly_hok/JIT.ex:3` manages runtime compilation:
  - stores AST/type/call-graph in a process (`module_server`) (`lib/poly_hok/JIT.ex:257`)
  - infers types (`lib/poly_hok/JIT.ex:158`)
  - does closure elimination (`lib/poly_hok/JIT.ex:463`)
  - emits kernel/device CUDA snippets (`compile_kernel`, `compile_function`)
- `lib/poly_hok/cuda_backend.ex:1` is the AST-to-CUDA codegen backend and module transformer.
- `lib/poly_hok/type_inference.ex:1` performs custom type inference/checking for DSL expressions/statements.

## Native layer

- `c_src/gpu_nifs.cu:1656` registers NIFs for `Elixir.PolyHok`.
- Key native entrypoints:
  - runtime JIT compile+launch via NVRTC: `jit_compile_and_launch_nif` (`c_src/gpu_nifs.cu:193`)
  - device memory alloc/copy: `new_gpu_array_nif` (`c_src/gpu_nifs.cu:734`), `create_gpu_array_nx_nif` (`c_src/gpu_nifs.cu:571`), `get_gpu_array_nif` (`c_src/gpu_nifs.cu:447`)
  - static kernel/function pointer loading + launch: `load_kernel_nif` (`c_src/gpu_nifs.cu:1461`), `load_fun_nif` (`c_src/gpu_nifs.cu:1532`), `spawn_nif` (`c_src/gpu_nifs.cu:1615`)

## How you use it (developer workflow)

- Build NIF shared object with `Makefile` (`priv/gpu_nifs.so` target).
- Write kernels/device fns in `PolyHok.defmodule ... do ... end`.
- Move tensors to GPU with `PolyHok.new_gnx`.
- Execute skeletons (`Ske.map`, `Ske.reduce`, etc.) optionally with `Fusion.with_fusion`.
- Pull results back with `PolyHok.get_gnx`.
- Real usage examples live in `experiments/*.exs` (for example `experiments/new_fusion.exs:1`, `experiments/softmax.exs:1`).
