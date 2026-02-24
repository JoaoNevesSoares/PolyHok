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

  defp benchmark(runs, fun) do
    start_time = System.monotonic_time()

    last_result =
      Enum.reduce(1..runs, nil, fn _, _ ->
        fun.()
      end)

    end_time = System.monotonic_time()
    elapsed_ms = System.convert_time_unit(end_time - start_time, :native, :millisecond)
    avg_ms = elapsed_ms / runs
    {last_result, elapsed_ms, avg_ms}
  end

  def run(n \\ 4_194_304, runs \\ 30) do
    :rand.seed(:exsplus, {5_347, 5_347, 5_347})

    host_x = Dp.random_tensor(n, -50.0, 100.0)
    host_y = Dp.random_tensor(n, -50.0, 100.0)

    dev_x = PolyHok.new_gnx(host_x)
    dev_y = PolyHok.new_gnx(host_y)

    unfused_fun = fn ->
      Ske.map2(dev_x, dev_y, &Dp.mult/2)
      |> Ske.reduce(0.0, &Dp.sum/2)
      |> PolyHok.get_gnx
    end

    fused_fun = fn ->
      Fusion.with_fusion(
        Ske.map2(dev_x, dev_y, &Dp.mult/2)
        |> Ske.reduce(0.0, &Dp.sum/2)
      )
      |> PolyHok.get_gnx
    end

    # Warmup to reduce one-time JIT/compilation overhead in timing output.
    _ = unfused_fun.()
    _ = fused_fun.()

    {unfused_res, unfused_ms, unfused_avg} = benchmark(runs, unfused_fun)
    {fused_res, fused_ms, fused_avg} = benchmark(runs, fused_fun)

    diff =
      unfused_res
      |> Nx.subtract(fused_res)
      |> Nx.abs()
      |> Nx.to_flat_list()
      |> hd()

    speedup =
      if fused_ms > 0 do
        unfused_ms / fused_ms
      else
        :infinity
      end

    IO.inspect(unfused_res, label: "Unfused result (last run)")
    IO.puts(
      "PolyHok.Unfused\t#{n}\truns=#{runs}\ttotal=#{unfused_ms} ms\tavg=#{Float.round(unfused_avg, 3)} ms"
    )

    IO.inspect(fused_res, label: "Fused result (last run)")
    IO.puts("PolyHok.Fused\t#{n}\truns=#{runs}\ttotal=#{fused_ms} ms\tavg=#{Float.round(fused_avg, 3)} ms")

    IO.puts("Speedup (unfused/fused): #{speedup}")
    IO.puts("Absolute difference (last run): #{diff}")
  end
end

args = Enum.reject(System.argv(), &(&1 == "--"))

n =
  case args do
    [arg_n | _] -> String.to_integer(arg_n)
    _ -> 4_194_304
  end

runs =
  case args do
    [_arg_n, arg_runs | _] -> String.to_integer(arg_runs)
    _ -> 30
  end

DotBenchmark.run(n, runs)
