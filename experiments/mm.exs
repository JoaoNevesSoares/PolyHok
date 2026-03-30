require PolyHok

PolyHok.defmodule MM do
  use Ske

  defd dot_loop(mat1, mat2, m, x, y) do
    tmp = 0.0

    for i in range(0, m, 1) do
      tmp = tmp + mat1[x * m + i] * mat2[i * m + y]
    end

    return(tmp)
  end

  defk map2xy2D_kernel(arr1, arr2, par, resp, size, f) do
    row = blockIdx.y * blockDim.y + threadIdx.y
    col = blockIdx.x * blockDim.x + threadIdx.x

    if(col < size && row < size) do
      resp[row * size + col] = f(arr1, arr2, par, row, col)
    end
  end

  def map2xy2D1p(arr1, arr2, par, resp, size, f) do
    block_size = 16
    grid_rows = trunc((size + block_size - 1) / block_size)
    grid_cols = trunc((size + block_size - 1) / block_size)

    PolyHok.spawn(&MM.map2xy2D_kernel/6, {grid_cols, grid_rows, 1}, {block_size, block_size, 1}, [
      arr1,
      arr2,
      par,
      resp,
      size,
      f
    ])
  end

  def comp2xy2D1p(arr1, arr2, par, size1, size2, f) do
    result_gpu = PolyHok.new_gnx(size1, size2, PolyHok.get_array_type(arr1))
    arr1_gpu = PolyHok.new_gnx(arr1)
    arr2_gpu = PolyHok.new_gnx(arr2)

    MM.map2xy2D1p(arr1_gpu, arr2_gpu, par, result_gpu, size1, f)

    r_gpu = PolyHok.get_gnx(result_gpu)
    r_gpu
  end

  def test(vector) do
    res =
      PolyHok.new_gnx(vector)
# Constatado que nested parallelism não da pra fazer
|> Ske.map(PolyHok.phok fn x -> Ske.map(x, PolyHok.phok(fn y -> y * 2 end)) end)
      |> PolyHok.get_gnx()
    IO.inspect(res)
  end
end

m = 2

mat1 = Nx.tensor(Enum.to_list(1..(m * m)), type: :f32)
mat2 = Nx.tensor(Enum.to_list(1..(m * m)), type: :f32)
n1 = Nx.tensor(Enum.to_list(1..10), type: :f32)

mat1 = Nx.reshape(mat1, {m, m})
mat2 = Nx.reshape(mat2, {m, m})

IO.inspect(mat1)
IO.inspect(mat2)

result =
  MM.comp2xy2D1p(mat1, mat2, m, m, m, &MM.dot_loop/4)

IO.inspect(result)
MM.test(n1)
