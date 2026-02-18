require PolyHok

PolyHok.defmodule Axpy do

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

defmodule BenchFusion do
  require PolyHok
  require Fusion
  require Axpy
  use Ske

  def run_normal() do
    n = 1048576

    :rand.seed(:exsplus, {5_347, 5_347, 5_347})

    host_x = Axpy.random_tensor(n, 0.0, 1.0)
    host_y = Axpy.random_tensor(n, 0.0, 1.0)
    host_z = Axpy.random_tensor(n, 0.0, 1.0)

    dev_x = PolyHok.new_gnx(host_x)
    dev_y = PolyHok.new_gnx(host_y)
    dev_z = PolyHok.new_gnx(host_z)

    prev = System.monotonic_time()
    dev_z =
      dev_x
      |> Ske.map2(dev_y, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.1) end))
      |> Ske.map2(dev_z, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.3) end))

    next = System.monotonic_time()
    host_z = PolyHok.get_gnx(dev_z)

    IO.inspect(host_z)
    IO.puts("PolyHok\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)} ms")
  end

  def run_fusion() do
    n = 1048576
    :rand.seed(:exsplus, {5_347, 5_347, 5_347})
    host_x = Axpy.random_tensor(n, 0.0, 1.0)
    host_y = Axpy.random_tensor(n, 0.0, 1.0)
    host_z = Axpy.random_tensor(n, 0.0, 1.0)

    dev_x = PolyHok.new_gnx(host_x)
    dev_y = PolyHok.new_gnx(host_y)
    dev_z = PolyHok.new_gnx(host_z)
    prev = System.monotonic_time()

    dev_z =
      Fusion.with_fusion(
        Ske.map2(dev_x, dev_y, PolyHok.phok(fn x, y -> Axpy.axpy(x, y, 0.1) end))
        |> Ske.map2(dev_z, PolyHok.phok(fn y, z -> Axpy.axpy(y, z, 0.3) end))
      )

    next = System.monotonic_time()

    host_z = PolyHok.get_gnx(dev_z)

    IO.inspect(host_z)
    IO.puts("PolyHok\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)} ms")
  end
end

# BenchFusion.run_normal()
BenchFusion.run_fusion()
