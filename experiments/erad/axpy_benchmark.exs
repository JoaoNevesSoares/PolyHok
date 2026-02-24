require PolyHok

PolyHok.defmodule Axpy do
  defd axpy(x, y, a) do
    a * x + y
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

defmodule AxpyBenchmark do
  require Fusion
  require Axpy
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
    host_x = Axpy.random_tensor(n, 0.0, 1.0)
    host_y = Axpy.random_tensor(n, 0.0, 1.0)
    host_z = Axpy.random_tensor(n, 0.0, 1.0)

    dev_x = PolyHok.new_gnx(host_x)
    dev_y = PolyHok.new_gnx(host_y)
    dev_z = PolyHok.new_gnx(host_z)

    unfused_fun = fn ->
      Ske.map2(dev_x, dev_y, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.1) end))
      |> Ske.map2(dev_z, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.3) end))
    end

    fused_fun = fn ->
      Fusion.with_fusion(
        Ske.map2(dev_x, dev_y, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.1) end))
        |> Ske.map2(dev_y, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.3) end))
      )
    end

    # warmup JIT/Compilation
    _ = unfused_fun.()
    # warmup JIT/Compilation
    _ = fused_fun.()

    {unfused_res, unfused_ms, unfused_avg} = benchmark(runs, unfused_fun)
    {fused_res, fused_ms, fused_avg} = benchmark(runs, fused_fun)

    speedup =
      if fused_ms > 0 do
        unfused_ms / fused_ms
      else
        :infinity
      end

    IO.puts(
      "PolyHok.Unfused\t#{n}\truns=#{runs}\ttotal=#{unfused_ms} ms\tavg=#{Float.round(unfused_avg, 3)} ms"
    )
    IO.puts("PolyHok.Fused\t#{n}\truns=#{runs}\ttotal=#{fused_ms} ms\tavg=#{Float.round(fused_avg, 3)} ms")

    IO.puts("Speedup (unfused/fused): #{speedup}")
  end
end

AxpyBenchmark.run(16_777_216, 30)
