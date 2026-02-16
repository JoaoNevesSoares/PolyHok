require PolyHok

PolyHok.defmodule PortfolioPricingExp do
  include(CAS_Poly)

  defd norm_cdf(x) do
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    rsqrt_2pi = 0.3989422804014327
    p = 0.2316419

    ax = fabsf(x)
    k = 1.0 / (1.0 + p * ax)
    poly = (((((a5 * k + a4) * k + a3) * k + a2) * k + a1) * k)
    phi = rsqrt_2pi * expf(-0.5 * ax * ax)
    cnd = 1.0 - phi * poly

    if x < 0.0 do
      cnd = 1.0 - cnd
    end

    return cnd
  end

  defd blackscholes_body(s, x, t, r, v) do
    sqrt_t = sqrtf(t)
    d1 = (logf(s / x) + (r + (v * v) / 2.0) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    return s * norm_cdf(d1) - x * expf(-r * t) * norm_cdf(d2)
  end

  defk black_scholes_weighted_kernel(out_contrib, stock, strike, years, weight, riskfree, volatility, n) do
    tid = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    i = tid

    while i < n do
      call_price = blackscholes_body(stock[i], strike[i], years[i], riskfree, volatility)
      out_contrib[i] = call_price * weight[i]
      i = i + stride
    end
  end

  defk reduce_sum_kernel(in_arr, out_scalar, initial, n, cas) do
    __shared__(cache[256])

    tid = threadIdx.x + blockIdx.x * blockDim.x
    cache_index = threadIdx.x
    stride = blockDim.x * gridDim.x

    temp = initial

    while tid < n do
      temp = in_arr[tid] + temp
      tid = tid + stride
    end

    cache[cache_index] = temp
    __syncthreads()

    i = blockDim.x / 2

    while i != 0 do
      if cache_index < i do
        cache[cache_index] = cache[cache_index + i] + cache[cache_index]
      end

      __syncthreads()
      i = i / 2
    end

    if cache_index == 0 do
      current_value = out_scalar[0]

      while !(current_value == cas(out_scalar, current_value, cache[0] + current_value)) do
        current_value = out_scalar[0]
      end
    end
  end
end

defmodule PortfolioPricingExperiment do
  @opt_n 1048576 
  @riskfree 0.02
  @volatility 0.30
  @threads_per_block 256
  @abs_tol 1.0e-4
  @rel_tol 1.0e-6

  def run do
    # :rand.seed(:exsplus, {5_347, 5_347, 5_347})
    :rand.seed(:exsplus, {1_347, 2_347, 3_347})

    stock = random_tensor(@opt_n, 5.0, 30.0)
    strike = random_tensor(@opt_n, 1.0, 100.0)
    years = random_tensor(@opt_n, 0.25, 10.0)
    weight = random_tensor(@opt_n, -2.0, 2.0)

    stock_gpu = PolyHok.new_gnx(stock)
    strike_gpu = PolyHok.new_gnx(strike)
    years_gpu = PolyHok.new_gnx(years)
    weight_gpu = PolyHok.new_gnx(weight)

    contrib_gpu = run_weighted_kernel(stock_gpu, strike_gpu, years_gpu, weight_gpu, @riskfree, @volatility)
    portfolio_gpu = run_reduce_kernel(contrib_gpu)

    stock_list = Nx.to_flat_list(stock)
    strike_list = Nx.to_flat_list(strike)
    years_list = Nx.to_flat_list(years)
    weight_list = Nx.to_flat_list(weight)

    portfolio_cpu =
      stock_list
      |> Enum.zip(strike_list)
      |> Enum.zip(years_list)
      |> Enum.zip(weight_list)
      |> Enum.reduce(0.0, fn {{{s, x}, t}, w}, acc ->
        acc + black_scholes_body_cpu(s, x, t, @riskfree, @volatility) * w
      end)

    abs_err = abs(portfolio_gpu - portfolio_cpu)
    rel_err = abs_err / max(abs(portfolio_cpu), 1.0e-12)

    IO.puts("The total value of the call options portfolio is: " <> format_float(portfolio_gpu))
    IO.puts("CPU reference portfolio value: " <> format_float(portfolio_cpu))
    IO.puts("Absolute error: " <> format_float(abs_err))
    IO.puts("Relative error: " <> format_float(rel_err))

    if abs_err <= @abs_tol or rel_err <= @rel_tol do
      IO.puts("Test passed.")
    else
      IO.puts("Test failed!")
    end
  end

  defp run_weighted_kernel(stock_gpu, strike_gpu, years_gpu, weight_gpu, riskfree, volatility) do
    ensure_same_shape!([stock_gpu, strike_gpu, years_gpu, weight_gpu], "run_weighted_kernel")
    ensure_f32!([stock_gpu, strike_gpu, years_gpu, weight_gpu], "run_weighted_kernel")

    shape = PolyHok.get_shape_gnx(stock_gpu)
    size = Tuple.product(shape)
    contrib_gpu = PolyHok.new_gnx(shape, {:f, 32})

    blocks = min(div(size + @threads_per_block - 1, @threads_per_block), 65_535)

    PolyHok.spawn(
      &PortfolioPricingExp.black_scholes_weighted_kernel/8,
      {blocks, 1, 1},
      {@threads_per_block, 1, 1},
      [contrib_gpu, stock_gpu, strike_gpu, years_gpu, weight_gpu, riskfree, volatility, size]
    )

    contrib_gpu
  end

  defp run_reduce_kernel(contrib_gpu) do
    ensure_f32!([contrib_gpu], "run_reduce_kernel")

    shape = PolyHok.get_shape_gnx(contrib_gpu)
    size = Tuple.product(shape)
    blocks = min(div(size + @threads_per_block - 1, @threads_per_block), 65_535)

    out_gpu = PolyHok.new_gnx(Nx.tensor([[0.0]], type: {:f, 32}))
    cas = PolyHok.phok(fn x, y, z -> cas_float(x, y, z) end)

    PolyHok.spawn(
      &PortfolioPricingExp.reduce_sum_kernel/5,
      {blocks, 1, 1},
      {@threads_per_block, 1, 1},
      [contrib_gpu, out_gpu, 0.0, size, cas]
    )

    out_gpu
    |> PolyHok.get_gnx()
    |> Nx.squeeze()
    |> Nx.to_number()
  end

  defp ensure_same_shape!([h | t], context) do
    shape = PolyHok.get_shape_gnx(h)

    Enum.each(t, fn tensor ->
      if PolyHok.get_shape_gnx(tensor) != shape do
        raise "#{context}: input shapes must match"
      end
    end)
  end

  defp ensure_f32!(tensors, context) do
    Enum.each(tensors, fn tensor ->
      if PolyHok.get_type_gnx(tensor) != {:f, 32} do
        raise "#{context}: only {:f, 32} tensors are supported"
      end
    end)
  end

  defp random_tensor(n, low, high) do
    vals =
      for _ <- 1..n do
        t = :rand.uniform()
        (1.0 - t) * low + t * high
      end

    Nx.tensor(vals, type: {:f, 32})
  end

  defp cnd(d) do
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    rsqrt_2pi = 0.3989422804014327

    k = 1.0 / (1.0 + 0.2316419 * abs(d))

    cnd =
      rsqrt_2pi * :math.exp(-0.5 * d * d) *
        (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))))

    if d > 0.0, do: 1.0 - cnd, else: cnd
  end

  defp black_scholes_body_cpu(s, x, t, r, v) do
    sqrt_t = :math.sqrt(t)
    d1 = (:math.log(s / x) + (r + 0.5 * v * v) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    s * cnd(d1) - x * :math.exp(-r * t) * cnd(d2)
  end

  defp format_float(v), do: :erlang.float_to_binary(v, [{:scientific, 6}])
end

PortfolioPricingExperiment.run()
