require PolyHok
require Integer

PolyHok.defmodule Ni do
  defd compute_xi(i, a, h) do
    type(out(float))
    out = a + i * h
    return(out)
  end
end

defmodule NiBenchmark do
  require Fusion
  require PolyHokInspect
  require Ni
  use Ske

  defp benchmark(runs \\ 30, fun) do
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

  def run(runs \\ 30) do
    sub = 2 * 1_048_575
    i_list = Enum.to_list(0..sub)

    w_list =
      Enum.map(i_list, fn
        0 -> 1
        ^sub -> 1
        x when Integer.is_even(x) -> 2
        _ -> 4
      end)

    i = Nx.tensor(i_list, type: :f32)
    w = Nx.tensor(w_list, type: :f32)

    i_gpu = PolyHok.new_gnx(i)
    w_gpu = PolyHok.new_gnx(w)

    unfused_fun = fn ->
      Ske.map(
        i_gpu,
        PolyHok.phok(fn i ->
          a = 0.0
          b = 3.1415926535
          sub = 2.0 * 1_048_575.0
          h = (b - a) / sub
          Ni.compute_xi(i, a, h)
        end)
      )
      |> Ske.map(PolyHok.phok(fn xi -> sinf(xi) end))
      |> Ske.map2(w_gpu, PolyHok.phok(fn fx, wi -> fx * wi end))
      |> Ske.reduce(0.0, PolyHok.phok(fn x, acc -> acc + x end))
      |> Ske.map(
        PolyHok.phok(fn s ->
          a = 0.0
          b = 3.1415926535
          sub = 2.0 * 1_048_575.0
          h = (b - a) / sub
          s * (h / 3.0)
        end)
      )
      |> PolyHok.get_gnx()
    end

    fused_fun = fn ->
      Fusion.with_fusion(
        Ske.map(
          i_gpu,
          PolyHok.phok(fn i ->
            a = 0.0
            b = 3.1415926535
            sub = 2.0 * 1_048_575.0
            h = (b - a) / sub
            a + i * h
          end)
        )
        |> Ske.map(PolyHok.phok(fn xi -> sinf(xi) end))
        |> Ske.map2(w_gpu, PolyHok.phok(fn fx, wi -> fx * wi end))
        |> Ske.reduce(0.0, PolyHok.phok(fn x, acc -> acc + x end))
      )
      |> Ske.map(
        PolyHok.phok(fn s ->
          a = 0.0
          b = 3.1415926535
          sub = 2.0 * 1_048_575.0
          h = (b - a) / sub
          s * (h / 3.0)
        end)
      )
      |> PolyHok.get_gnx()
    end

    # fused_fun = fn ->
    #   Fusion.with_fusion(
    #     Ske.map2(dev_x, dev_y, &Dp.mult/2)
    #     |> Ske.reduce(0.0, &Dp.sum/2)
    #   )
    #   |> PolyHok.get_gnx()
    # end

    # Warmup to reduce one-time JIT/compilation overhead in timing output.

    _ = unfused_fun.()
    _ = fused_fun.()

    {unfused_res, unfused_ms, unfused_avg} = benchmark(30, unfused_fun)

    {fused_res, fused_ms, fused_avg} = benchmark(30, fused_fun)

    # diff =
    #   unfused_res
    #   |> Nx.subtract(fused_res)
    #   |> Nx.abs()
    #   |> Nx.to_flat_list()
    #   |> hd()

    speedup =
      if fused_ms > 0 do
        unfused_ms / fused_ms
      else
        :infinity
      end

    # IO.inspect(res, label: "Unfused result (last run)")

    IO.puts(
      "PolyHok.Unfused\truns=#{runs}\ttotal=#{unfused_ms} ms\tavg=#{Float.round(unfused_avg, 3)} ms"
    )

    IO.inspect(fused_res, label: "Fused result (last run)")

    IO.puts(
      "PolyHok.Fused\truns=#{runs}\ttotal=#{fused_ms} ms\tavg=#{Float.round(fused_avg, 3)} ms"
    )

    IO.puts("Speedup (unfused/fused): #{speedup}")
    # IO.puts("Absolute difference (last run): #{diff}")
  end
end

NiBenchmark.run()
