require PolyHok
import BoundAnalysis

PolyHok.defmodule SimpleTest do
  defk saxpy(n, a, x, y) do
    i = blockIdx.x * blockDim.x + threadIdx.x

    if(i < n) do
      y[i] = a * x[i] + y[i]
    end
  end

  defk hell(n, a, y, z) do
    i = blockIdx.x * blockDim.x + threadIdx.x

    if(i < n) do
      z[i] = a * y[i]
    end
  end
end

x_cpu = Nx.tensor(Enum.to_list(1..10), type: {:f, 32})
y_cpu = Nx.tensor(Enum.to_list(1..10), type: {:f, 32})
z_cpu = Nx.tensor(Enum.to_list(1..10), type: {:f, 32})

x_gpu = x_cpu |> PolyHok.new_gnx()
y_gpu = y_cpu |> PolyHok.new_gnx()
z_gpu = z_cpu |> PolyHok.new_gnx()

fuse(
  PolyHok.spawn(
    &SimpleTest.saxpy/4,
    {1, 1, 1},
    {10, 1, 1},
    [10, 2.5, x_gpu, y_gpu]
  ),
  PolyHok.spawn(
    &SimpleTest.hell/4,
    {1, 1, 1},
    {10, 1, 1},
    [10, 2.0, y_gpu, z_gpu]
  )
)

# result = PolyHok.get_gnx(y_gpu)
# IO.inspect(result, label: "Result after kernel execution")
