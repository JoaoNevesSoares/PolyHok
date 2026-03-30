require PolyHok

PolyHok.defmodule Teste do
  defd adiciona(x_val) do
    x_val + 1
  end

  defk map_ker(a1, a2, size, f) do
    index = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index, size, stride) do
      a2[i] = f(a1[i])
    end
  end

  defk hello_ker(a, n) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if(tid < n) do
      a[tid] = 1.0
    end
  end

  def map(input, f) do
    shape = PolyHok.get_shape(input)
    type = PolyHok.get_type(input)

    result_gpu = PolyHok.new_gnx(shape, type)
    size = Tuple.product(shape)

    threadsPerBlock = 128
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(
      &Teste.map_ker/4,
      {numberOfBlocks, 1, 1},
      {threadsPerBlock, 1, 1},
      [input, result_gpu, size, f]
    )

    result_gpu
  end

  def teste_fun(vec_a) do
    res = map(vec_a, &Teste.adiciona/1)
  end
end

h_arr = Nx.tensor(Enum.to_list(1..10), type: {:f, 32})
d_arr = PolyHok.new_gnx(h_arr)
d_res = Teste.teste_fun(d_arr)
h_res = PolyHok.get_gnx(d_res)
IO.inspect(h_res)
