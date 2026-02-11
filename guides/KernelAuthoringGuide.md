# Kernel Authoring Guide

This guide summarizes the kernel DSL that PolyHok compiles to CUDA, based on the current implementation in `lib/poly_hok/cuda_backend.ex`, `lib/poly_hok/type_inference.ex`, and the example kernels in `lib/poly_hok/ske.ex`.

## Quick Start

Define a module with GPU kernels using `PolyHok.defmodule`, then launch with `PolyHok.spawn`.

```elixir
require PolyHok

PolyHok.defmodule MyKernels do
  defk map2_kernel(a1, a2, out, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    if id < size do
      out[id] = f(a1[id], a2[id])
    end
  end
end

threads = 256
blocks = div(size + threads - 1, threads)
PolyHok.spawn(&MyKernels.map2_kernel/5, {blocks, 1, 1}, {threads, 1, 1}, [a1, a2, out, size, f])
```

You can pass GPU-callable lambdas using `PolyHok.phok/1`:

```elixir
f = PolyHok.phok(fn x, y -> x + y end)
```

## Module-Level DSL

- `PolyHok.defmodule Name do ... end`
  - Declares a GPU module that can contain kernels and device functions.
- `defk name(args...) do ... end`
  - Defines a CUDA kernel.
- `defd name(args...) do ... end`
  - Defines a device/helper function callable from kernels.
- `deft name(type_signature)`
  - Defines an explicit type signature for a kernel or device function.
- `include(ModName)`
  - Inlines CUDA code from `c_src/Elixir.ModName.cu` into the generated module.
- `PolyHok.phok(fn ... -> ... end)`
  - Defines an anonymous GPU function that can be passed as an argument to kernels.
- `PolyHok.clo(fn ... -> ... end)`
  - Defines a GPU closure (captures free vars). Used when you need free variables in device functions.

## Kernel Body DSL Forms

This is the complete set of statement and expression forms that the CUDA backend recognizes.

### Statements

- Assignment
  - `x = expr`
  - `arr[idx] = expr`
- If / else
  - `if cond do ... end`
  - `if cond do ... else ... end`
- While
  - `while cond do ... end`
- Do-while
  - `do_while do ... end`
  - `do_while_test cond`
- For-range
  - `for i in range(n) do ... end`
  - `for i in range(start, stop) do ... end`
  - `for i in range(start, stop, step) do ... end`
- Shared memory
  - `__shared__(buf[size])`
- Local array declaration (float array)
  - `buf[size]`
- Typed variable declaration
  - `var x = float 0.0`
  - `var x = int 0`
  - `var x = double 0.0`
  - `var x float`
  - `var x int`
  - `var x double`
- Type annotation (affects inference, not emitted in CUDA)
  - `type x float`
  - `type x int`
  - `type x double`
- Return
  - `return(expr)`
- Function call as a statement
  - `f(a, b, c)`

### Expressions

- Array access
  - `arr[idx]`
- Struct/field access
  - `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `gridDim.x`
- Binary and unary operators
  - `+`, `-`, `*`, `/`, `<=`, `<`, `>`, `>=`, `&&`, `||`, `!`, `!=`, `==`
- Variable reference
  - `x`
- Function call
  - `f(a, b, c)`
- Literals
  - integers, floats, strings

## Notes on Types and Inference

- The compiler tracks variable types with a simple inference pass. Variables must be declared before use.
- Use `var` or `type` forms if inference can’t determine a type, or when you need to force a type.
- `__shared__` declarations infer their element type from the declared variable type.

## Common Intrinsics

These are used as normal field accesses:

- `threadIdx.x`, `threadIdx.y`, `threadIdx.z`
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z`
- `blockDim.x`, `blockDim.y`, `blockDim.z`
- `gridDim.x`, `gridDim.y`, `gridDim.z`

Synchronization is a normal function call:

- `__syncthreads()`

## Examples in Tree

- `lib/poly_hok/ske.ex`: real kernels using `defk`, shared memory, `range/3`, and GPU lambdas.
- `experiments/*.exs`: device function composition using `defd` and AST construction.
