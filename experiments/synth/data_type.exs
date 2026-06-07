require PolyHok

PolyHok.defmodule Dp do
  require PolyHokInspect
  require Fusion

  defd mult(x) do
    x * 3.0
  end

  defd mult_2(x) do
    x * 1.6
  end

  defd square(u) do
    u * u
  end

  defd div(g) do
    g / 2.0
  end

  defd rsqr(h) do
    sqrtf(h)
  end

  defd add(y) do
    y + 1
  end

  defd add_2(t) do
    t + 0.43
  end

  defd sub(z) do
    z - 0.29 
  end

  defd inv(t) do
    if t != 0.0 do
      t * 1.0
    else
      t + 1.0
    end
  end

  def random_tensor(n, low, high) do
    vals =
      for _ <- 1..n do
        t = :rand.uniform()
        (1.0 - t) * low + t * high
      end

    # IO.inspect(vals, label: "vals")
    Nx.tensor(vals, type: {:f, 64})
  end

  def run_unfused() do
    n = 100_000_000
    host_x = Dp.random_tensor(n, -10.0, 10.0)
    dev_x = PolyHok.new_gnx(host_x)

    prev = System.monotonic_time()

    res =
      Ske.map(dev_x, &Dp.mult/1)
      |> Ske.map(&Dp.sub/1)
      |> Ske.map(&Dp.div/1)
      |> PolyHok.get_gnx()

    next = System.monotonic_time()
    t = PolyHok.get_gnx(dev_x)

    IO.puts("Run unfused")
    IO.puts("PolyHok\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)} ms")
  end

  def run_fused() do
    n = 100_000_000
    host_x = Dp.random_tensor(n, -10.0, 10.0)
    dev_x = PolyHok.new_gnx(host_x)

    prev = System.monotonic_time()

    res =
      PolyHokInspect.block_inspect do
        Fusion.with_fusion(
          Ske.map(dev_x, &Dp.mult/1)
          |> Ske.map(&Dp.sub/1)
          |> Ske.map(&Dp.div/1)
        )
      end
      |> PolyHok.get_gnx()

    next = System.monotonic_time()
    t = PolyHok.get_gnx(dev_x)
    IO.puts("Run fused")
    # IO.inspect(res, label: "res")
    # IO.inspect(t, label: "dev_x")
    IO.puts("PolyHok\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)} ms")
  end
end

require Fusion
use Ske

Enum.each(1..30, fn execution ->
  IO.puts("Execucao: #{execution}")
  # Dp.run_unfused()
  Dp.run_fused()
end)
