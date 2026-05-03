require PolyHok
import Fusion
require PolyHokInspect

PolyHok.defmodule Hello do
  defk hello(input) do
    tid = threadIdx.x
    input[tid] = tid + 1
  end

  defk world(input) do
    tid = threadIdx.x
    res = input[tid]
  end
end

host = Nx.tensor(Enum.to_list(1..10), type: :s32)
gpu = PolyHok.new_gnx(host)

PolyHok.spawn(&Hello.hello/1, {1, 1, 1}, {10, 1, 1}, [gpu])
<~> PolyHok.spawn(&Hello.world/1, {1, 1, 1}, {10, 1, 1}, [gpu])

res_d = PolyHok.get_gnx(gpu)
IO.inspect(res_d)
