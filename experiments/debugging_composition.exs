require PolyHok

PolyHok.defmodule MyModule do
  defd inc(x) do
    x + 1
  end

  defd mult(y) do
    y * 2
  end
end

defmodule Example do
  require PolyHok
  require PolyHokInspect
  use Ske
  require MyModule
  require Fusion

  def main() do
    n = 5
    arr1 = Nx.tensor(Enum.to_list(1..n), type: {:s, 32})
    gpu_arr = PolyHok.new_gnx(arr1)
    gpu_res = Fusion.with_fusion(Ske.map(gpu_arr, &MyModule.mult/1) |> Ske.map(&MyModule.inc/1))
    # gpu_res = PolyHokInspect.block_inspect do
    #   Ske.map(gpu_arr, &MyModule.mult/1) |> Ske.map(&MyModule.inc/1)
    # end

    # gpu_res =
    #   Ske.map(
    #     gpu_arr,
    #     PolyHok.phok(fn x ->
    #       type y int
    #       y = mult(x)
    #       return inc(y)
    #     end)
    #   )
    host_res = PolyHok.get_gnx(gpu_res)
    IO.inspect(host_res)
  end
end

Example.main()
