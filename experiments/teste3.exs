require PolyHok

PolyHok.defmodule Teste do

  defk hello(a, n) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if(tid < n) do
      a[tid] = 1.0 
    end
  end

  def launch(input) do
    shape = PolyHok.get_shape(input)
    size = Tuple.product(shape)

    threadsPerBlock = 128
    numberOfBlocks = 1

    PolyHok.spawn(
      &Teste.hello/2,
      {numberOfBlocks, 1, 1},
      {threadsPerBlock, 1, 1},
      [input, size]
    )
    input
  end
end

h_arr = Nx.tensor(Enum.to_list(1..128), type: {:f, 32})
d_arr = PolyHok.new_gnx(h_arr)
d_res = Teste.launch(d_arr)
h_res = PolyHok.get_gnx(d_res)
IO.inspect(h_res)
