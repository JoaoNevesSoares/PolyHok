require PolyHok

PolyHok.defmodule CompareKernels do
  defd mult(x, a) do
    x * a
  end

  defd add(x, b) do
    x + b
  end

  defd square(x) do
    x * x
  end
end

defmodule ChainedMapFusionCompare do
  require PolyHok
  require Fusion
  require CompareKernels
  use Ske

  defp scalar_like_gnx(value, like_gnx) when is_number(value) do
    type = PolyHok.get_type_gnx(like_gnx)
    shape = PolyHok.get_shape_gnx(like_gnx)

    value
    |> Nx.tensor(type: type)
    |> Nx.broadcast(shape)
    |> PolyHok.new_gnx()
  end

  defp random_tensor(n, low, high, seed \\ 1234) do
    key = Nx.Random.key(seed)
    {t, _key2} = Nx.Random.uniform(key, low, high, shape: {n}, type: {:f, 32})
    t
  end

  defp non_fused_chain(x, a, b) do
    Ske.map2(x, a, &CompareKernels.mult/2)
    |> Ske.map2(b, &CompareKernels.add/2)
    |> Ske.map(&CompareKernels.square/1)
    |> Ske.map2(b, &CompareKernels.add/2)
  end

  defp fused_chain(x, a, b) do
    Fusion.with_fusion(
      Ske.map2(x, a, &CompareKernels.mult/2)
      |> Ske.map2(b, &CompareKernels.add/2)
      |> Ske.map(&CompareKernels.square/1)
      |> Ske.map2(b, &CompareKernels.add/2)
    )
  end

  defp report_diff(label, lhs, rhs) do
    diff = Nx.subtract(lhs, rhs)
    abs_diff = Nx.abs(diff)
    l2 = Nx.sqrt(Nx.sum(Nx.multiply(diff, diff)))
    linf = Nx.reduce_max(abs_diff)

    IO.puts("#{label}")
    IO.puts("  L2:   #{Nx.to_number(l2)}")
    IO.puts("  Linf: #{Nx.to_number(linf)}")
  end

  def run(n \\ 1_000_000) do
    x_host = random_tensor(n, -2.0, 3.0)
    x = PolyHok.new_gnx(x_host)
    a = scalar_like_gnx(0.75, x)
    b = scalar_like_gnx(0.15, x)

    ref =
      x_host
      |> Nx.multiply(0.75)
      |> Nx.add(0.15)
      |> then(fn y -> Nx.multiply(y, y) end)
      |> Nx.add(0.15)

    # warmup to reduce one-time kernel/JIT overhead in timing output
    _ = non_fused_chain(x, a, b) |> PolyHok.get_gnx()
    _ = fused_chain(x, a, b) |> PolyHok.get_gnx()

    {non_fused_us, non_fused_host} =
      :timer.tc(fn ->
        non_fused_chain(x, a, b) |> PolyHok.get_gnx()
      end)

    {fused_us, fused_host} =
      :timer.tc(fn ->
        fused_chain(x, a, b) |> PolyHok.get_gnx()
      end)

    IO.puts("n=#{n}")
    IO.puts("Non-fused time: #{Float.round(non_fused_us / 1000.0, 3)} ms")
    IO.puts("Fused time:     #{Float.round(fused_us / 1000.0, 3)} ms")

    speedup =
      if fused_us > 0 do
        non_fused_us / fused_us
      else
        :infinity
      end

    IO.puts("Speedup (non_fused / fused): #{speedup}")

    report_diff("Diff: fused vs non-fused", fused_host, non_fused_host)
    report_diff("Diff: fused vs reference", fused_host, ref)
    report_diff("Diff: non-fused vs reference", non_fused_host, ref)
  end
end

n =
  case Enum.reject(System.argv(), &(&1 == "--")) do
    [arg | _] ->
      String.to_integer(arg)

    _ ->
      1_000_000
  end

ChainedMapFusionCompare.run(n)
