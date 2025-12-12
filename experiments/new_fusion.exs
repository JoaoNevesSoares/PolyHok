require PolyHok

PolyHok.defmodule MyModule do
  defd sum(a, b) do
    a + b
  end
  defd acc(x) do
    x * 2
  end
  defd double(c) do
    x = c * c
    return x
  end

  defd mult(c, d) do
    x = c * d
    return x
  end
end

defmodule Example do
  require PolyHok
  require PolyHokInspect
  require MyModule
  require Fusion
  use Ske

#   def map2map2() do
#     a = PolyHok.new_gnx(Nx.tensor([1, 2, 3], type: {:s, 32}))
#     b = PolyHok.new_gnx(Nx.tensor([10, 20, 30], type: {:s, 32}))
#     c = PolyHok.new_gnx(Nx.tensor([100, 200, 300], type: {:s, 32}))

#     gpu_res = PolyHokInspect.block_inspect do
#           Fusion.with_fusion(Ske.map2(a, b, &MyModule.sum/2) |> Ske.map2(c, &MyModule.mult/2))
#     end
#     host_res = PolyHok.get_gnx(gpu_res)
#     IO.inspect(host_res, label: "map2 |> map2")
# end

# def mapmap() do
#   a = PolyHok.new_gnx(Nx.tensor([1, 2, 3], type: {:s, 32}))
#   gpu_res = PolyHokInspect.block_inspect do
#           Fusion.with_fusion(Ske.map(a, PolyHok.phok(fn x -> x + 1 end)) |> Ske.map(&MyModule.acc/1))
#     end
#     host_res = PolyHok.get_gnx(gpu_res)
#     IO.inspect(host_res, label: "map |> map")
# end

  def mapReduce() do
    a = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})

    gpu_a = PolyHok.new_gnx(a)
    gpu_res = PolyHokInspect.block_inspect do
    Fusion.with_fusion(Ske.map(gpu_a, PolyHok.phok(fn x -> return x + 1 end)) |> Ske.reduce(0.0, &MyModule.sum/2))
    end
    host_res =
  gpu_res
  |> PolyHok.get_gnx()

  IO.inspect(host_res, label: "map2 |> reduce")
  end

  # def map2Reduce() do
  #   a = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})
  #   b = Nx.tensor([10.0, 20.0, 30.0], type: {:f, 32})

  #   gpu_a = PolyHok.new_gnx(a)
  #   gpu_b = PolyHok.new_gnx(b)
  #   gpu_res = PolyHokInspect.block_inspect do
  #   Fusion.with_fusion(Ske.map2(gpu_a, gpu_b, &MyModule.mult/2) |> Ske.reduce(0.0, &MyModule.sum/2))
  #   end
  #   host_res =
  # gpu_res
  # |> PolyHok.get_gnx()

  # IO.inspect(host_res, label: "map2 |> reduce")
  # end
end

# Example.mapmap()
Example.mapReduce()
# Example.map2map2()
# Example.map2Reduce()
