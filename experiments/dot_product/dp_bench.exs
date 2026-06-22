Code.require_file("dp.exs", __DIR__)

defmodule Dpbench do
  def run(:fused, x, y) do
    start_time = System.monotonic_time()
    res = Dp.dot_fs(x, y)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def run(:unfused, x, y) do
    start_time = System.monotonic_time()
    res = Dp.dot_unfs(x, y)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def main() do
    argv = System.argv()

    {[{:size, n}, {:type, type} | _rest], [], []} =
      OptionParser.parse(argv, strict: [size: :integer, type: :string])

    file = File.open!("#{n}_dot.csv", [:write, :utf8])

    dev_a = PolyHok.random_gnx(0, 1, n)
    dev_b = PolyHok.random_gnx(0, 1, n)

    res =
      Enum.reduce(1..30, [], fn _, acc ->
        fs = run(:fused, dev_a, dev_b)
        ufs = run(:unfused, dev_a, dev_b)

        acc ++ [%{"unfused" => ufs, "fused" => fs}]
      end)
    res
    |> CSV.encode(headers: ["unfused", "fused"])
    |> Enum.each(&IO.write(file, &1))

    File.close(file)
  end
end

Dpbench.main()
