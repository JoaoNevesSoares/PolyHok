require PolyHok

PolyHok.defmodule Clotest do
  defd mult(x, a) do
    x * a
  end
end

defmodule Runtest do
  require PolyHok
  require Clotest
  use Ske

  def run() do
    h_arr = Nx.tensor(Enum.to_list(1..10), type: {:f, 32})
    d_arr = PolyHok.new_gnx(h_arr)
    a = 2.0

    func = PolyHok.clo(fn x -> Clotest.mult(x, a) end)

    gpu_res = Ske.map(d_arr, func)
    host_res = PolyHok.get_gnx(gpu_res)
    IO.inspect(host_res)
  end
end

Runtest.run()
