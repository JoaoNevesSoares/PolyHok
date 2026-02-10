require PolyHok

PolyHok.defmodule MyModule do
  defd mult_a(x) do
    x * 2
  end
  defd mult_b(y) do
    y * 3
  end
end
defmodule Example do
  require PolyHok
  require PolyHokInspect
  use Ske
  require MyModule
  require Fusion
  def main() do
    n = 100_000_000
    arr1 = Nx.tensor(Enum.to_list(1..n), type: {:s, 32})
    gpu_arr = PolyHok.new_gnx(arr1)
    gpu_res = Fusion.with_fusion(Ske.map(gpu_arr, &MyModule.mult_a/1) |> Ske.map(&MyModule.mult_b/1))
    host_res = PolyHok.get_gnx(gpu_res)
    IO.inspect(host_res)
  end
end

Example.main()
