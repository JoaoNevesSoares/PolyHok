require PolyHok

PolyHok.defmodule DevFunctions do
  defd mult(xs, a) do
    a * xs
  end

  defd square(xs) do
    xs * xs
  end

  defd add(xs, ys) do
    xs + ys
  end
end

defmodule UnfusedBench do
  require PolyHok
  require DevFunctions
  use Ske

  defp problem_size(n) do
    Integer.pow(2, n)
  end

  def gen_inputs() do
    size = problem_size(28) 
    h_arr = Nx.tensor([Enum.to_list(1..size)],type: {:f, 32})
    IO.inspect(h_arr)
    d_arr = PolyHok.new_gnx(h_arr)
    IO.inspect(d_arr)
  end

  def map_map() do
    h_arr = Nx.tensor([Enum.to_list(1..10)], type: {:f, 32})
    A = PolyHok.new_gnx(h_arr)
res = Ske.map(A, PolyHok.phok fn x -> x + 1 end)

    host_res = PolyHok.get_gnx(res)
    IO.inspect(host_res)
  end
end

UnfusedBench.map_map()
