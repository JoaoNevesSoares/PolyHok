require PolyHok
import Fusion
require PolyHokInspect

PolyHok.defmodule Hello do
  defk hello(input) do
    tid = threadIdx.x
    input[tid] = tid + 10
  end

  defk world(output) do
    tid = threadIdx.y
    output[tid] = tid
  end
end

host = Nx.tensor(Enum.to_list(1..10), type: :s32)
host_d = Nx.tensor(List.duplicate(0,10), type: :s32)
gpu = PolyHok.new_gnx(host)
gpu_d = PolyHok.new_gnx(host_d)

PolyHok.spawn(&Hello.hello/1, {1, 1, 1}, {10, 1, 1}, [gpu])  <~> PolyHok.spawn(&Hello.world/1, {1, 1, 1}, {1, 10, 1}, [gpu_d])

res_d = PolyHok.get_gnx(gpu_d)
# IO.inspect(res)
IO.inspect(res_d)
