require PolyHok

PolyHok.defmodule MyModule do
  defd mult(x, a) do
    x * a
  end

  defd sum(y1, b) do
    y1 + b
  end

  defd square(y2) do
    y2 * y2
  end

  defd bias(y3, c) do
    y3 + c
  end

  defd clamp(y4, ax, bx) do
    type(t(float))
    type(z(float))
    t = fmaxf(y4, ax)
    z = fminf(t, bx)
    return(z)
  end
end

defmodule Example do
  require PolyHokInspect
  require PolyHok
  use Ske
  require MyModule
  require Fusion 

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

  defp report_error_stats(gpu_tensor, ref_tensor) do
    diff = Nx.subtract(gpu_tensor, ref_tensor)
    abs_diff = Nx.abs(diff)
    l1 = Nx.sum(abs_diff)
    l2 = Nx.sqrt(Nx.sum(Nx.multiply(diff, diff)))
    linf = Nx.reduce_max(abs_diff)

    ref_12 = Nx.sqrt(Nx.sum(Nx.multiply(ref_tensor, ref_tensor)))
    eps = Nx.tensor(1.0e-12, type: {:f, 32})
    rel_l2 = Nx.divide(l2, Nx.add(ref_12, eps))

    ref_max = Nx.reduce_max(Nx.abs(ref_tensor))
    IO.puts("Error metrics")
    IO.puts("  L1(abs):   #{Nx.to_number(l1)}")
    IO.puts("  L2:        #{Nx.to_number(l2)}")
    IO.puts("  Linf:      #{Nx.to_number(linf)}")
    IO.puts("  rel L2:    #{Nx.to_number(rel_l2)}")
    IO.puts("  max|ref|:  #{Nx.to_number(ref_max)}")
  end

  def run_three_chain(input) do
    
    x = random_tensor(input, -2.0, 3.0) |> PolyHok.new_gnx()
    a = scalar_like_gnx(0.5, x)
    b = scalar_like_gnx(0.1, x)
    c = scalar_like_gnx(-0.2, x)

    gpu_res =
      PolyHokInspect.block_inspect do 
        Fusion.with_fusion(Ske.map2(x, a, &MyModule.mult/2) |> Ske.map2(b, &MyModule.sum/2) |> Ske.map(&MyModule.square/1))
      end
    res = PolyHok.get_gnx(gpu_res)
    IO.inspect(res)
  end

  def run(input) do
    x_host = random_tensor(input, -2.0, 3.0)
    x = random_tensor(input, -2.0, 3.0) |> PolyHok.new_gnx()
    a = scalar_like_gnx(0.5, x)
    b = scalar_like_gnx(0.1, x)
    c = scalar_like_gnx(-0.2, x)
    ax = scalar_like_gnx(0.0, x)
    bx = scalar_like_gnx(1.0, x)

    gpu_res =
      Ske.map2(x, a, &MyModule.mult/2)
      |> Ske.map2(b, &MyModule.sum/2)
      |> Ske.map(&MyModule.square/1)
      |> Ske.map2(c, &MyModule.bias/2)
      |> Ske.map3(ax, bx, &MyModule.clamp/3)

    ref =
      x_host
      |> Nx.multiply(0.5)
      |> Nx.add(0.1)
      |> then(fn y2 -> Nx.multiply(y2, y2) end)
      |> Nx.add(-0.2)
      |> Nx.max(0.0)
      |> Nx.min(1.0)

  gpu_host = PolyHok.get_gnx(gpu_res)
  report_error_stats(gpu_host, ref)
  IO.inspect(gpu_host)
    IO.inspect(ref)
  end
end

Example.run_three_chain(512)
