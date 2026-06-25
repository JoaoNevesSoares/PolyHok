Code.require_file("dp.exs", __DIR__)

defmodule Dpbench do
  def run(:fused, x, y, initial) do
    start_time = System.monotonic_time()
    res = Dp.dot_fs(x, y, initial)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def run(:unfused, x, y, initial) do
    start_time = System.monotonic_time()
    res = Dp.dot_unfs(x, y, initial)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def gen_data(low, high, n, type) do
    a = PolyHok.random_gnx(low, high, n, type)
    b = PolyHok.random_gnx(low, high, n, type)
    [a, b]
  end

  def main() do
    argv = System.argv()

    {[{:size, n}, {:type, type}, {:order, ord} | _rest], [], []} =
      OptionParser.parse(argv, strict: [size: :integer, type: :string, order: :integer])

    file = File.open!("results/dot/dot_#{type}_#{ord}_#{n}.csv", [:write, :utf8])

    [dev_a, dev_b, initial] =
      case type do
        "float" -> gen_data(0, 1, n, {:f, 32}) ++ [0.0]
        "double" -> gen_data(0, 1, n, {:f, 64}) ++ [0.0]
        "integer" -> gen_data(0, 2, n, {:s, 32}) ++ [0]
      end

    start = ord
    finish = 29 + ord
    res =
      Enum.reduce(start..finish, [], fn i, acc ->
        if rem(i, 2) == 0 do
          fs = run(:fused, dev_a, dev_b,initial)
          ufs = run(:unfused, dev_a, dev_b, initial)
          acc ++ [%{"unfused" => ufs, "fused" => fs}]
        else
          ufs = run(:unfused, dev_a, dev_b, initial)
          fs = run(:fused, dev_a, dev_b, initial)
          acc ++ [%{"unfused" => ufs, "fused" => fs}]
        end
      end)

    res
    |> CSV.encode(headers: ["unfused", "fused"])
    |> Enum.each(&IO.write(file, &1))

    File.close(file)
  end
end

Dpbench.main()
