Code.require_file("axpy.exs", __DIR__)

defmodule Axpybench do
  def run(:fused, a, b, c) do
    start_time = System.monotonic_time()
    res = Axpy.fused(a, b, c)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def run(:unfused, a, b, c) do
    start_time = System.monotonic_time()
    res = Axpy.unfused(a, b, c)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def gen_data(low, high, n, type) do
    a = PolyHok.random_gnx(low, high, n, type)
    b = PolyHok.random_gnx(low, high, n, type)
    c = PolyHok.random_gnx(low, high, n, type)
    {a, b, c}
  end

  def main() do
    argv = System.argv()

    {[{:size, n}, {:type, type}, {:order, ord} | _rest], [], []} =
      OptionParser.parse(argv, strict: [size: :integer, type: :string, order: :integer])

    file = File.open!("results/axpy/axpy_#{type}_#{n}_#{ord}_axpy.csv", [:write, :utf8])

    {dev_a, dev_b, dev_c} =
      case type do
        "float" -> gen_data(0, 10, n, {:f, 32})
        "double" -> gen_data(0, 10, n, {:f, 64})
        "integer" -> gen_data(0, 10, n, {:s, 32})
      end

    start = ord
    finish = 29 + ord
    res =
      Enum.reduce(start..finish, [], fn i, acc ->
        if rem(i, 2) == 0 do
          fs = run(:fused, dev_a, dev_b, dev_c)
          ufs = run(:unfused, dev_a, dev_b, dev_c)
          acc ++ [%{"unfused" => ufs, "fused" => fs}]
        else
          ufs = run(:unfused, dev_a, dev_b, dev_c)
          fs = run(:fused, dev_a, dev_b, dev_c)
          acc ++ [%{"unfused" => ufs, "fused" => fs}]
        end
      end)

    res
    |> CSV.encode(headers: ["unfused", "fused"])
    |> Enum.each(&IO.write(file, &1))

    File.close(file)
  end
end

Axpybench.main()
