require PolyHok
use Ske

PolyHok.defmodule Teste do
  def teste_fun(vec_a) do
    res = Enum.reduce(0..10, vec_a, fn _x, acc -> 
      t = Ske.map(acc, PolyHok.phok(fn y -> y + y end)) 
      rs = PolyHok.get_gnx(t)
      IO.inspect(rs)
      t
    end)
    res
  end
end

h_arr = Nx.tensor(Enum.to_list(1..10), type: {:f, 32})
d_arr = PolyHok.new_gnx(h_arr)
d_res = Teste.teste_fun(d_arr)
h_res = PolyHok.get_gnx(d_res)
IO.inspect(h_res)
