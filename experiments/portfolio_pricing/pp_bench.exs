Code.require_file("pp.exs", __DIR__)

defmodule Ppbench do
  def run(:fused, stock, strike, years, weight, volatility, riskfree) do
    start_time = System.monotonic_time()
    res = PortfolioPricing.run_fs(stock, strike, years, weight, volatility, riskfree)
    end_time = System.monotonic_time()
    gpu = PolyHok.get_gnx(res)
    gpu_res = Nx.to_number(gpu[0][0]) |> IO.inspect(label: "fused_res")
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def run(:unfused, stock, strike, years, weight, volatility, riskfree) do
    start_time = System.monotonic_time()
    res = PortfolioPricing.run_unfs(stock, strike, years, weight, volatility, riskfree)
    end_time = System.monotonic_time()
    gpu = PolyHok.get_gnx(res)
    gpu_res = Nx.to_number(gpu[0][0]) |> IO.inspect(label: "unfused_res")
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def main() do
    argv = System.argv()

    {[{:size, n} | _rest], [], []} =
      OptionParser.parse(argv, strict: [size: :integer])

    file = File.open!("#{n}_pp_2.csv", [:write, :utf8])

    stock = PolyHok.random_gnx(5, 30, n)
    strike = PolyHok.random_gnx(1, 100, n)
    years = PolyHok.random_gnx(1, 10, n)
    weight = PolyHok.random_gnx(-2, 2, n)
    volatility = 0.3
    riskfree = 0.02

    res =
      Enum.reduce(1..30, [], fn _, acc ->
        fs = run(:fused, stock, strike, years, weight, volatility, riskfree)
        ufs = run(:unfused, stock, strike, years, weight, volatility, riskfree)

        acc ++ [%{"unfused" => ufs, "fused" => fs}]
      end)

    res
    |> CSV.encode(headers: ["unfused", "fused"])
    |> Enum.each(&IO.write(file, &1))

    File.close(file)
  end
end

Ppbench.main()
