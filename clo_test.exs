require PolyHok
require PolyHokInspect
use Ske

x = 10
gpu_gnx = PolyHok.random_gnx(1, 2, 32)
res = PolyHokInspect.block_inspect do 
Ske.map(gpu_gnx, PolyHok.phok(fn t -> t + 1 end))
end

IO.inspect(PolyHok.get_gnx(res), label: "result: ")
