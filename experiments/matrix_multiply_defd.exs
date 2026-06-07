require PolyHok

PolyHok.defmodule MatrixMultiplyKernels do
  defd multiply_add(a_value, b_value, acc) do
    return(acc + a_value * b_value)
  end

  defk matmul_kernel(a, b, c, rows_a, cols_a, cols_b) do
    __shared__(tile_a[256])
    __shared__(tile_b[256])

    local_row = threadIdx.y
    local_col = threadIdx.x
    row = blockIdx.y * blockDim.y + threadIdx.y
    col = blockIdx.x * blockDim.x + threadIdx.x
    tile_index = local_row * 16 + local_col
    sum = 0.0

    for tile in range(0, cols_a, 16) do
      a_col = tile + local_col
      b_row = tile + local_row

      if row < rows_a && a_col < cols_a do
        tile_a[tile_index] = a[row * cols_a + a_col]
      else
        tile_a[tile_index] = 0.0
      end

      if b_row < cols_a && col < cols_b do
        tile_b[tile_index] = b[b_row * cols_b + col]
      else
        tile_b[tile_index] = 0.0
      end

      __syncthreads()

      for k in range(0, 16, 1) do
        sum = multiply_add(tile_a[local_row * 16 + k], tile_b[k * 16 + local_col], sum)
      end

      __syncthreads()
    end

    if row < rows_a && col < cols_b do
      c[row * cols_b + col] = sum
    end
  end
end

defmodule MatrixMultiplyExperiment do
  @tile_size 16
  @tolerance 1.0e-4

  def run do
    a =
      Nx.tensor(
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]
        ],
        type: {:f, 32}
      )

    b =
      Nx.tensor(
        [
          [7.0, 8.0],
          [9.0, 10.0],
          [11.0, 12.0]
        ],
        type: {:f, 32}
      )

    result = matmul(a, b)
    expected = Nx.dot(a, b)

    IO.inspect(a, label: "A")
    IO.inspect(b, label: "B")
    IO.inspect(result, label: "A x B on GPU")
    IO.inspect(expected, label: "A x B on CPU")

    max_delta = max_abs_delta(expected, result)
    IO.puts("Max absolute error: " <> :erlang.float_to_binary(max_delta, [{:scientific, 6}]))

    if max_delta > @tolerance do
      raise "matrix multiplication result differs from the CPU reference"
    end

    result
  end

  def matmul(a, b) do
    {rows_a, cols_a} = Nx.shape(a)
    {rows_b, cols_b} = Nx.shape(b)

    unless cols_a == rows_b do
      raise "matmul: left columns must match right rows"
    end

    unless Nx.type(a) == {:f, 32} and Nx.type(b) == {:f, 32} do
      raise "matmul: this experiment expects {:f, 32} tensors"
    end

    a_gpu = PolyHok.new_gnx(a)
    b_gpu = PolyHok.new_gnx(b)
    c_gpu = PolyHok.new_gnx({rows_a, cols_b}, {:f, 32})

    blocks_x = div(cols_b + @tile_size - 1, @tile_size)
    blocks_y = div(rows_a + @tile_size - 1, @tile_size)

    PolyHok.spawn(
      &MatrixMultiplyKernels.matmul_kernel/6,
      {blocks_x, blocks_y, 1},
      {@tile_size, @tile_size, 1},
      [a_gpu, b_gpu, c_gpu, rows_a, cols_a, cols_b]
    )

    PolyHok.get_gnx(c_gpu)
  end

  defp max_abs_delta(expected, actual) do
    expected
    |> Nx.subtract(actual)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end
end

MatrixMultiplyExperiment.run()
