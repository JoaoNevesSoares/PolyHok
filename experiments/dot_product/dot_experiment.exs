require PolyHok

PolyHok.defmodule Dp do
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

defmodule DotBenchmark do
  require Fusion
  require Dp
  use Ske

  def run_unfused(n) do
    :rand.seed(:exsplus, {5_347, 5_347, 5_347})

    host_x = Dp.random_tensor(n, -50.0, 100.0)
    host_y = Dp.random_tensor(n, -50.0, 100.0)

    dev_x = PolyHok.new_gnx(host_x)
    dev_y = PolyHok.new_gnx(host_y)

    start_time = System.monotonic_time()

    res =
      Ske.map2(dev_x, dev_y, &Dp.mult/2)
      |> Ske.reduce(0.0, &Dp.sum/2)
        |> PolyHok.get_gnx()

    end_time = System.monotonic_time()

    elapsed_ms = System.convert_time_unit(end_time - start_time, :native, :millisecond)
    IO.puts("the total runtime is #{elapsed_ms}/ms")
    IO.inspect(res, label: "vector result")
  end

  def run_fused(n) do
    :rand.seed(:exsplus, {5_347, 5_347, 5_347})

    host_x = Dp.random_tensor(n, -50.0, 100.0)
    host_y = Dp.random_tensor(n, -50.0, 100.0)

    dev_x = PolyHok.new_gnx(host_x)
    dev_y = PolyHok.new_gnx(host_y)

    start_time = System.monotonic_time()

    res =
      Fusion.with_fusion(
        Ske.map2(dev_x, dev_y, &Dp.mult/2)
        |> Ske.reduce(0.0, &Dp.sum/2)
      )
        |> PolyHok.get_gnx()

    end_time = System.monotonic_time()
    elapsed_ms = System.convert_time_unit(end_time - start_time, :native, :millisecond)
    IO.puts("the total runtime is #{elapsed_ms}/ms")
    IO.inspect(res, label: "vector result")
  end
end

n = 8388608
DotBenchmark.run_fused(n)
DotBenchmark.run_fused(n)
DotBenchmark.run_fused(n)
DotBenchmark.run_fused(n)
DotBenchmark.run_fused(n)
# DotBenchmark.run_unfused(n)
# DotBenchmark.run_unfused(n)
# DotBenchmark.run_unfused(n)
# DotBenchmark.run_unfused(n)
# DotBenchmark.run_unfused(n)
