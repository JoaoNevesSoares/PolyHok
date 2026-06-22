require PolyHok
use Ske

PolyHok.defmodule PortfolioPricing do
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
    poly = ((((a5 * k + a4) * k + a3) * k + a2) * k + a1) * k
    phi = rsqrt_2pi * expf(-0.5 * ax * ax)
    cnd = 1.0 - phi * poly

    if x < 0.0 do
      cnd = 1.0 - cnd
    end

    return(cnd)
  end

  defd blackscholes_body(s, x, t, r, v) do
    sqrt_t = sqrtf(t)
    d1 = (logf(s / x) + (r + v * v / 2.0) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    return(s * norm_cdf(d1) - x * expf(-r * t) * norm_cdf(d2))
  end

  defk black_scholes_weighted_kernel(
         out_contrib,
         stock,
         strike,
         years,
         weight,
         riskfree,
         volatility,
         n
       ) do
    tid = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    i = tid

    while i < n do
      call_price = blackscholes_body(stock[i], strike[i], years[i], riskfree, volatility)
      out_contrib[i] = call_price * weight[i]
      i = i + stride
    end
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

  def run_cpu(n) do
    riskfree = 0.02
    volatility = 0.30
    stock = PolyHok.random_gnx(5, 30, n) |> PolyHok.get_gnx() |> Nx.to_flat_list()
    strike = PolyHok.random_gnx(1, 100, n) |> PolyHok.get_gnx() |> Nx.to_flat_list()
    years = PolyHok.random_gnx(1, 10, n) |> PolyHok.get_gnx() |> Nx.to_flat_list()
    weight = PolyHok.random_gnx(-2, 2, n) |> PolyHok.get_gnx() |> Nx.to_flat_list()

    stock
    |> Enum.zip(strike)
    |> Enum.zip(years)
    |> Enum.zip(weight)
    |> Enum.reduce(0.0, fn {{{s, x}, t}, w}, acc ->
      acc + black_scholes_body_cpu(s, x, t, riskfree, volatility) * w
    end)
  end

  # def run_unfs(n) do
  #   riskfree = 0.02
  #   volatility = 0.30
  # end

  def run_spawn(n) do
    riskfree = 0.02
    volatility = 0.30
    threads_per_block = 256
    abs_tol = 1.0e-4
    rel_tol = 1.0e-6

    stock = PolyHok.random_gnx(5, 30, n)
    strike = PolyHok.random_gnx(1, 100, n)
    years = PolyHok.random_gnx(1, 10, n)
    weight = PolyHok.random_gnx(-2, 2, n)

    shape = PolyHok.get_shape_gnx(stock)
    size = Tuple.product(shape)
    contrib = PolyHok.new_gnx(shape, {:f, 32})
    blocks = min(div(size + threads_per_block - 1, threads_per_block), 65_535)

    PolyHok.spawn(
      &PortfolioPricing.black_scholes_weighted_kernel/8,
      {blocks, 1, 1},
      {threads_per_block, 1, 1},
      [contrib, stock, strike, years, weight, riskfree, volatility, size]
    )

    out_portfolio_price =
      Ske.reduce(contrib, 0.0, PolyHok.phok(fn x, acc -> x + acc end))
      |> PolyHok.get_gnx()
  end

  defp format_float(v), do: :erlang.float_to_binary(v, [{:scientific, 6}])

  def compare_gpu_vs_cpu do
    n = 1_048_576
    abs_tol = 1.0e-4
    rel_tol = 1.0e-6

    gpu = run_spawn(n) 
    gpu_res = Nx.to_number(gpu[0][0]) |> IO.inspect(label: "gpu_res")
    cpu_res = run_cpu(n) |> IO.inspect(label: "cpu_res")
    abs_err = abs(gpu_res - cpu_res)
    rel_err = abs_err / max(abs(cpu_res), 1.0e-12)
    IO.puts("The total value of the call options portfolio is: " <> format_float(gpu_res))
    IO.puts("CPU reference portfolio value: " <> format_float(cpu_res))
    IO.puts("Absolute error: " <> format_float(abs_err))
    IO.puts("Relative error: " <> format_float(rel_err))

    if abs_err <= abs_tol or rel_err <= rel_tol do
      IO.puts("Test passed.")
    else
      IO.puts("Test failed!")
    end
  end
end

PortfolioPricing.compare_gpu_vs_cpu()
