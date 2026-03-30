require PolyHok

PolyHok.defmodule Nested do
  # define a kernel, i.e., a function that will execute entirely on the GPU
  defk hello_kernel(in_arr, out_arr) do
    tid = threadIdx.x
    out_arr[tid] = in_arr[tid]
  end

  defk map(in_arr, fun) do

    tid = threadIdx.x;
    res = fun(in_arr[i])
  end

  defd map(input_arr, fun) do

  end

  def matmul(A, B) do

    map(A, fn a_row ->
      map(B, fn ->
        b_col
        some_function_f(a_row, b_col)
      end)
    end)
  end

  # define a host entry point for computation
  def main() do
    threadsPerBlock = {10, 1, 1}
    numberOfBlocks = {1, 1, 1}

    h_arr = Nx.tensor(Enum.to_list(1..10), type: {:s, 32})
    d_arr = PolyHok.new_gnx(h_arr)
    shape = PolyHok.get_shape(d_arr)
    type = PolyHok.get_type(d_arr)
    d_res = PolyHok.new_gnx(shape, type)

    PolyHok.spawn(
      &Hello.hello_kernel/2,
      numberOfBlocks,
      threadsPerBlock,
      [d_arr, d_res]
    )

    PolyHok.get_gnx(d_res)
  end
end

res = Hello.main()
IO.inspect(res)
