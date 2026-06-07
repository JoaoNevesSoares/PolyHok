require PolyHok
require Integer
use Ske

PolyHok.defmodule Ni do
  defd compute_xi(i, a, h) do
    a + i * h
  end
end

n = 100
sub = 2 * n          # number of subintervals (must be even)
m = sub + 1          # number of points

a = 0.0
b = :math.pi()

h = (b - a) / sub

i_list = Enum.to_list(0..sub)

w_list =
  Enum.map(i_list, fn
    0 -> 1
    ^sub -> 1
    x when Integer.is_even(x) -> 2
    _ -> 4
  end)

# i can be float, but keep it consistent
i = Nx.tensor(i_list, type: :f32)
w = Nx.tensor(w_list, type: :f32)

i_gpu = PolyHok.new_gnx(i)
w_gpu = PolyHok.new_gnx(w)

sum =
  i_gpu
  |> Ske.map(
    PolyHok.phok(fn i ->
      Ni.compute_xi(i, 0.0, 0.0157075)
    end)
  )
  |> Ske.map(PolyHok.phok(fn xi -> sinf(xi) end))
  |> Ske.map2(w_gpu, PolyHok.phok(fn fx, wi -> fx * wi end))
  |> Ske.reduce(0.0, PolyHok.phok(fn x, acc -> acc + x end))

# multiply by h/3 after reduce
res =
  sum
  |> Ske.map(PolyHok.phok(fn s -> s * (0.0157075 / 3.0) end))

IO.inspect(PolyHok.get_gnx(res))
