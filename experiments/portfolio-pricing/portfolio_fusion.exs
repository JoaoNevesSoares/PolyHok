require PolyHok

PolyHok.defmodule PortfolioPricingExp do
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

  defd blackscholes_body(s, x, t) do
    sqrt_t = sqrtf(t)
    d1 = (logf(s / x) + (0.02 + 0.9 / 2.0) * t) / (0.3 * sqrt_t)
    d2 = d1 - 0.3 * sqrt_t
    return(s * norm_cdf(d1) - x * expf(-0.02 * t) * norm_cdf(d2))
  end

  defd weight_multiply(price, weight) do
    return(price * weight)
  end

  defd sum(opt, port) do
    opt + port
  end
end

defmodule PortfolioPricingExperiment do
  # @opt_n 10_000_000
  # @opt_n 20_000_000
  @opt_n 40_000_000

  use Ske
  require Fusion

  def run do
    :rand.seed(:exsplus, {1_347, 2_347, 3_347})

    stock = random_tensor(@opt_n, 5.0, 30.0)
    strike = random_tensor(@opt_n, 1.0, 100.0)
    years = random_tensor(@opt_n, 0.25, 10.0)
    weight = random_tensor(@opt_n, -2.0, 2.0)

    stock_gpu = PolyHok.new_gnx(stock)
    strike_gpu = PolyHok.new_gnx(strike)
    years_gpu = PolyHok.new_gnx(years)
    weight_gpu = PolyHok.new_gnx(weight)

    prev = System.monotonic_time()

    res =
      Ske.map3(stock_gpu, strike_gpu, years_gpu, &PortfolioPricingExp.blackscholes_body/3)
      |> Ske.map2(weight_gpu, &PortfolioPricingExp.weight_multiply/2)
      |> Ske.reduce(0.0, &PortfolioPricingExp.sum/2)
      |> PolyHok.get_gnx()

    next = System.monotonic_time()
    IO.inspect(res, label: "output price of portfolio")
    IO.puts("PolyHok\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
  end

  def run_fused() do 
    :rand.seed(:exsplus, {1_347, 2_347, 3_347})

    stock = random_tensor(@opt_n, 5.0, 30.0)
    strike = random_tensor(@opt_n, 1.0, 100.0)
    years = random_tensor(@opt_n, 0.25, 10.0)
    weight = random_tensor(@opt_n, -2.0, 2.0)

    stock_gpu = PolyHok.new_gnx(stock)
    strike_gpu = PolyHok.new_gnx(strike)
    years_gpu = PolyHok.new_gnx(years)
    weight_gpu = PolyHok.new_gnx(weight)

    prev = System.monotonic_time()

    res =
      Fusion.with_fusion(
      Ske.map3(stock_gpu, strike_gpu, years_gpu, &PortfolioPricingExp.blackscholes_body/3)
      |> Ske.map2(weight_gpu, &PortfolioPricingExp.weight_multiply/2)
      |> Ske.reduce(0.0, &PortfolioPricingExp.sum/2))
      |> PolyHok.get_gnx()

    next = System.monotonic_time()
    IO.inspect(res, label: "output price of portfolio")
    IO.puts("PolyHok\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
  end

  defp random_tensor(n, low, high) do
    vals =
      for _ <- 1..n do
        t = :rand.uniform()
        (1.0 - t) * low + t * high
      end

    Nx.tensor(vals, type: {:f, 32})
  end
end


Enum.each(1..10, fn x -> 
# PortfolioPricingExperiment.run()
PortfolioPricingExperiment.run_fused()
end)
