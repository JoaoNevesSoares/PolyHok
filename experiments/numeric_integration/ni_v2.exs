require PolyHok
require Integer
use Ske

PolyHok.defmodule Ni do
  defd compute_xi(i, a, h) do
    a + i * h
  end
end

n = 100
m = 2 * n + 1

i_list = Enum.to_list(0..(m - 1))

w_list =
  Enum.map(i_list, fn
    x when x == 0 -> 1
    x when x == 2 * n -> 1
    x when Integer.is_even(x) -> 2
    x when Integer.is_odd(x) -> 4
  end)

i = Nx.tensor(i_list, type: :f32)
w = Nx.tensor(w_list, type: :f32)
i_gpu = PolyHok.new_gnx(i)
w_gpu = PolyHok.new_gnx(w)

temp =
  i_gpu
  |> Ske.map(
    PolyHok.phok(fn i ->
      m = 100.0
      a = 0.0
      b = 3.14
      h = (b - a) / m
      Ni.compute_xi(i, a, h)
    end)
  )
  |> Ske.map(
    PolyHok.phok(fn xi ->
      sinf(xi)
    end)) 
  |> Ske.map2(w_gpu, PolyHok.phok( fn (fx, wi) -> 
    fx * wi
    end))
  |> Ske.reduce(0.0, PolyHok.phok( fn x, acc -> acc + x end))

res = PolyHok.get_gnx(temp)
IO.inspect(res)
