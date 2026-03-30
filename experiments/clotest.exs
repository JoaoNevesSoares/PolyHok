require PolyHok

PolyHok.defmodule Clotest do
  defd mult(x, a) do
    t = 0.0
    t = x * a + t
    return t
  end
end

defmodule Runtest do
  require PolyHok
  require Clotest
  use Ske

  def run() do
    h_arr = Nx.tensor(Enum.to_list(1..10), type: {:f, 32})
    d_arr = PolyHok.new_gnx(h_arr)

    func = PolyHok.clo(fn i -> Clotest.mult(i,d_arr[0]) end)

    gpu_res = Ske.map(d_arr, func)
    host_res = PolyHok.get_gnx(gpu_res)
    IO.inspect(host_res)
  end
end

Runtest.run()
