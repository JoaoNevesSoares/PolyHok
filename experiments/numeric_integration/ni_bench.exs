Code.require_file("ni.exs", __DIR__)

defmodule Nibench do

def run(:fused) do
start_time = System.monotonic_time() 
res = Ni.ni_fs()
end_time = System.monotonic_time()
IO.inspect(res)
System.convert_time_unit(end_time - start_time, :native, :millisecond)
end

def run(:unfused) do
start_time = System.monotonic_time()
res = Ni.ni_unfs()
end_time = System.monotonic_time()
IO.inspect(res)
System.convert_time_unit(end_time - start_time, :native, :millisecond)
end

def main() do
argv = System.argv()

{[{:size, n}, {:type, type} | _rest], [], []} =
  OptionParser.parse(argv, strict: [size: :integer, type: :string])

file = File.open!("#{n}_ni.csv", [:write, :utf8])

res =
  Enum.reduce(1..30, [], fn _, acc ->
    fs = run(:fused)
    ufs = run(:unfused)

    acc ++ [%{"unfused" => ufs, "fused" => fs}]
  end)

  res 
  |> CSV.encode(headers: ["unfused", "fused"])
  |> Enum.each(&IO.write(file, &1))

File.close(file)

end
end

Nibench.main()
