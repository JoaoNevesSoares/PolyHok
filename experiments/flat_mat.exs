require PolyHok

PolyHok.defmodule MM do
  use Ske

  def matmul(a, b) do

Ske.map2(a, b, PolyHok.phok( fn x,y -> 
    for i in range(0,2, 1) do
      sum = x[i] * y[i]
    end
      return sum
    end
))

  end
end

m = 2

mat1 = Nx.tensor(Enum.to_list(1..(m * m)), type: :s32)
mat2 = Nx.tensor(Enum.to_list(1..(m * m)), type: :s32)

mat1 = Nx.reshape(mat1, {m, m})
mat2 = Nx.reshape(mat2, {m, m})
mat2 = Nx.transpose(mat2)

IO.inspect(mat1)
IO.inspect(mat2)
a = PolyHok.new_gnx(mat1)
b = PolyHok.new_gnx(mat2)

res = MM.matmul(a, b)
c = PolyHok.get_gnx(res)
IO.inspect(c)

