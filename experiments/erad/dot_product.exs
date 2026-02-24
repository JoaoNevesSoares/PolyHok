require PolyHok

PolyHok.defmodule Dp do
  # include(CAS_Double)

  defd mult(x, y) do
    x * y
  end
  defd sum(x, y) do
    x + y
  end

  def random_tensor(n, low, high) do
    vals =
      for _ <- 1..n do
        t = :rand.uniform()
        (1.0 - t) * low + t * high
      end

    Nx.tensor(vals, type: {:f, 32})
  end

end

use Ske

n = 4194304 

host_x = Dp.random_tensor(n, -50.0, 100.0)
host_y = Dp.random_tensor(n, -50.0, 100.0)

dev_x = PolyHok.new_gnx(host_x)
dev_y = PolyHok.new_gnx(host_y)

prev = System.monotonic_time()

res = Ske.map2(dev_x, dev_y, &Dp.mult/2)
  |> Ske.reduce(0.0, &Dp.sum/2)
  |> PolyHok.get_gnx
next = System.monotonic_time()
IO.inspect(res)
IO.puts("PolyHok\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)} ms")
