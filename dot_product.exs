require PolyHok

PolyHok.defmodule Dp do
  defd mult(x, y) do
    x * y
  end

  defd sum(x, y) do
    x + y
  end

  def random_tensor(n, low, high) do
    key = Nx.Random.key(System.os_time())

    {tensor, _new_key} =
      Nx.Random.uniform(
        key,
        low,
        high,
        shape: {n},
        type: {:f, 32}
      )

    tensor
  end
end

use Ske

n = 1000000

Enum.each(1..2, fn execution ->
  host_x = Dp.random_tensor(n, -1.0, 1.0)
  host_y = Dp.random_tensor(n, -1.0, 1.0)

  dev_x = PolyHok.new_gnx(host_x)
  dev_y = PolyHok.new_gnx(host_y)

  prev = System.monotonic_time()

  res =
    Ske.map2(dev_x, dev_y, &Dp.mult/2)
    |> Ske.reduce(0.0, &Dp.sum/2)
    |> PolyHok.get_gnx()

  next = System.monotonic_time()

  hx_dev = PolyHok.get_gnx(dev_x)
  hy_dev = PolyHok.get_gnx(dev_y)
  IO.puts("Execucao #{execution}")
  IO.puts("PolyHok\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)} ms")
end)
