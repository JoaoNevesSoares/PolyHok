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

def main() do
argv = System.argv()

{[{:size, n}, {:type, type} | _rest], [], []} =
  OptionParser.parse(argv, strict: [size: :integer, type: :string])

file = File.open!("#{n}_axpy.csv", [:write, :utf8])

dev_a = PolyHok.random_gnx(0, 10, n)
dev_b = PolyHok.random_gnx(0, 10, n)
dev_c = PolyHok.random_gnx(0, 10, n)

res =
  Enum.reduce(1..30, [], fn _, acc ->
    fs = run(:fused, dev_a, dev_b, dev_c)
    ufs = run(:unfused, dev_a, dev_b, dev_c)

    acc ++ [%{"unfused" => ufs, "fused" => fs}]
  end)

  res 
  |> CSV.encode(headers: ["unfused", "fused"])
  |> Enum.each(&IO.write(file, &1))

File.close(file)

end
end

Axpybench.main()
