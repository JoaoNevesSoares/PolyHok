require PolyHok
use Ske

PolyHok.defmodule PMap do

  defd inc(x) do
    x+1
  end
  defd square(xs) do
    xs*xs
  end
  defd add(xs, ys) do
    xs+ys
  end
end

size = Integer.pow(2, 10)
arr1 = Nx.iota({size}, type: {:f, 32}) 

prev = System.monotonic_time()

host_res1 = arr1
    |> PolyHok.new_gnx
    |> Ske.map(&PMap.inc/1)
    |> Ske.map(&PMap.square/1) 
    |> PolyHok.get_gnx

next = System.monotonic_time()

IO.puts "PolyHok 2^28\t#{System.convert_time_unit(next-prev,:native,:millisecond)}/ms"
IO.inspect host_res1
