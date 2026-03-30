require PolyHok
require Fusion

PolyHok.defmodule NumbaFusionKernels do
  defd map_affine_ab(x) do
    type a float
    type b float
    a = 1.1
    b = 0.3
    return(a * x + b)
  end

  defd map_square(x) do
    return(x * x)
  end

  defd map_fast_tanh_like(x) do
    type av float
    av = fabsf(x)
    return(x / (1.0 + av))
  end

  defd map_affine_cd(x) do
    type c float
    type d float
    c = 0.9
    d = -0.2
    return(c * x + d)
  end

  defd map_relu(x) do
    type z float
    z = fmaxf(x, 0.0)
    return(z)
  end
end

defmodule NumbaFusionBenchmark do
  require PolyHok
  require Fusion
  require NumbaFusionKernels
  use Ske

  @default_n 4_194_304
  @default_warmup 1
  @default_iters 10
  @default_seed 0
  @dtype {:f, 32}

  def run(opts \\ []) do
    n = Keyword.get(opts, :n, @default_n)
    warmup = Keyword.get(opts, :warmup, @default_warmup)
    iters = Keyword.get(opts, :iters, @default_iters)
    seed = Keyword.get(opts, :seed, @default_seed)

    x_host = make_synthetic(n, seed)
    x_gpu = PolyHok.new_gnx(x_host)

    unfused_ms = time_gpu(fn -> non_fused_chain(x_gpu) end, warmup, iters)
    fused_ms = time_gpu(fn -> fused_chain(x_gpu) end, warmup, iters)

    unfused_host = non_fused_chain(x_gpu) |> PolyHok.get_gnx()
    fused_host = fused_chain(x_gpu) |> PolyHok.get_gnx()
    ref_host = nx_reference(x_host)

    diff_fused_unfused = max_abs_diff(fused_host, unfused_host)
    diff_fused_ref = max_abs_diff(fused_host, ref_host)

    IO.puts("N = #{n}, dtype=float32")

    IO.puts(
      "[1] map chain, unfused #{format_ms(unfused_ms)} ms, fused #{format_ms(fused_ms)} ms, speedup #{format_speedup(unfused_ms, fused_ms)}x"
    )

    IO.puts("max|fused - unfused| = #{format_diff(diff_fused_unfused)}")
    IO.puts("max|fused - reference| = #{format_diff(diff_fused_ref)}")
  end

  def run_from_argv(argv) do
    {n, warmup, iters} = parse_argv(argv)
    run(n: n, warmup: warmup, iters: iters)
  end

  defp non_fused_chain(x) do
    x
    |> Ske.map(&NumbaFusionKernels.map_affine_ab/1)
    |> Ske.map(&NumbaFusionKernels.map_square/1)
    |> Ske.map(&NumbaFusionKernels.map_fast_tanh_like/1)
    |> Ske.map(&NumbaFusionKernels.map_affine_cd/1)
    |> Ske.map(&NumbaFusionKernels.map_relu/1)
  end

  defp fused_chain(x) do
    Fusion.with_fusion(
      Ske.map(x, &NumbaFusionKernels.map_affine_ab/1)
      |> Ske.map(&NumbaFusionKernels.map_square/1)
      |> Ske.map(&NumbaFusionKernels.map_fast_tanh_like/1)
      |> Ske.map(&NumbaFusionKernels.map_affine_cd/1)
      |> Ske.map(&NumbaFusionKernels.map_relu/1)
    )
  end

  defp nx_reference(x) do
    x
    |> Nx.multiply(1.1)
    |> Nx.add(0.3)
    |> then(fn v -> Nx.multiply(v, v) end)
    |> then(fn v -> Nx.divide(v, Nx.add(1.0, Nx.abs(v))) end)
    |> Nx.multiply(0.9)
    |> Nx.add(-0.2)
    |> Nx.max(0.0)
  end

  defp max_abs_diff(lhs, rhs) do
    lhs
    |> Nx.subtract(rhs)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  defp make_synthetic(n, seed) do
    key = Nx.Random.key(seed)
    {x, _new_key} = Nx.Random.normal(key, 0.0, 1.0, shape: {n}, type: @dtype)
    x
  end

  defp time_gpu(fun, warmup, iters) when warmup >= 0 and iters > 0 do
    if warmup > 0 do
      for _ <- 1..warmup do
        _ = fun.()
        sync_gpu()
      end
    end

    t0 = System.monotonic_time()

    for _ <- 1..iters do
      _ = fun.()
      sync_gpu()
    end

    t1 = System.monotonic_time()
    elapsed_us = System.convert_time_unit(t1 - t0, :native, :microsecond)
    elapsed_us / 1000.0 / iters
  end

  defp sync_gpu do
    PolyHok.synchronize()
  end

  defp parse_argv(argv) do
    n = parse_positive_int(Enum.at(argv, 0), @default_n)
    warmup = parse_non_negative_int(Enum.at(argv, 1), @default_warmup)
    iters = parse_positive_int(Enum.at(argv, 2), @default_iters)
    {n, warmup, iters}
  end

  defp parse_positive_int(nil, default), do: default

  defp parse_positive_int(text, _default) do
    case Integer.parse(text) do
      {value, ""} when value > 0 -> value
      _ -> raise ArgumentError, "expected positive integer, got: #{inspect(text)}"
    end
  end

  defp parse_non_negative_int(nil, default), do: default

  defp parse_non_negative_int(text, _default) do
    case Integer.parse(text) do
      {value, ""} when value >= 0 -> value
      _ -> raise ArgumentError, "expected non-negative integer, got: #{inspect(text)}"
    end
  end

  defp format_ms(ms), do: :erlang.float_to_binary(ms, decimals: 3)

  defp format_speedup(unfused_ms, fused_ms) when fused_ms > 0.0 do
    :erlang.float_to_binary(unfused_ms / fused_ms, decimals: 2)
  end

  defp format_speedup(_unfused_ms, _fused_ms), do: "inf"

  defp format_diff(diff), do: :erlang.float_to_binary(diff, scientific: 6)
end

NumbaFusionBenchmark.run_from_argv(Enum.reject(System.argv(), &(&1 == "--")))
