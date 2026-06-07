require PolyHok
require Fusion
require PolyHokInspect
use Ske

host_a = Nx.tensor(Enum.to_list(1..4), type: {:s, 32})
host_b = Nx.tensor(Enum.to_list(1..4), type: {:s, 32})
host_c = Nx.tensor(Enum.to_list(1..10), type: {:s, 32})
in_a = PolyHok.new_gnx(host_a)
in_b = PolyHok.new_gnx(host_b)
in_c = PolyHok.new_gnx(host_b)

res =
  PolyHokInspect.block_inspect do
    Fusion.with_fusion(
      Ske.map(in_a, PolyHok.phok(fn x -> x * 2 end))
      |> Ske.map2(in_b, PolyHok.phok(fn x,y -> 
        t = (x - y)
        return t
      end))
      |> Ske.reduce(0, PolyHok.phok(fn x, acc -> 

        if x > 0 do 
          return acc + x 
        else
          return acc
        end

        end)))
  end
  |> PolyHok.get_gnx()

IO.inspect(res)
