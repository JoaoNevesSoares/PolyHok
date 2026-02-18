require PolyHok

PolyHok.defmodule Axpy do
  # include(CAS_Double)

  defd axpy(x, y, a) do
    a * x + y
  end

  def random_tensor(n, low, high) do
    vals =
      for _ <- 1..n do
        t = :rand.uniform()
        (1.0 - t) * low + t * high
      end

    Nx.tensor(vals, type: {:f, 32})
  end
end

use Ske

n = 2048

:rand.seed(:exsplus, {5_347, 5_347, 5_347})

host_x = Axpy.random_tensor(n, 0.0, 1.0)
host_y = Axpy.random_tensor(n, 0.0, 1.0)
host_z = Axpy.random_tensor(n, 0.0, 1.0)
host_w = Axpy.random_tensor(n, 0.0, 1.0)

dev_x = PolyHok.new_gnx(host_x)
dev_y = PolyHok.new_gnx(host_y)
dev_z = PolyHok.new_gnx(host_z)
dev_w = PolyHok.new_gnx(host_w)

# shape = PolyHok.get_shape(input)
# type = PolyHok.get_type(input)
# result_gpu = PolyHok.new_gnx(shape,type)

prev = System.monotonic_time()

# dev_x =
#   Ske.map2(
#     dev_x,
#     dev_y,
#     PolyHok.clo(fn x, y ->
#       a = 1.1
#       Axpy.axpy(x, y, a)
#     end)
#   )

# dev_x = Ske.map2(dev_x, dev_y, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 1.1) end))
dev_w =
  dev_x
  |> Ske.map2(dev_y, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.1) end))
  |> Ske.map2(dev_z, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.3) end))
  |> Ske.map2(dev_w, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.9) end))

next = System.monotonic_time()

host_z = PolyHok.get_gnx(dev_w)

IO.inspect(host_z)
IO.puts("PolyHok\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)} ms")
