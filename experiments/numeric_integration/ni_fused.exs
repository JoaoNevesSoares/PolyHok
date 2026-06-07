require PolyHok
require Integer

PolyHok.defmodule Ni do
  defd compute_xi(i, a, h) do
    out = a + i * h
    return(out)
  end
end

defmodule NiBenchmark do
  require Fusion
  use Ske

  def run_unfused() do
    sub = 2 * 1_048_575
    i_list = Enum.to_list(0..sub)

    w_list =
      Enum.map(i_list, fn
        0 -> 1
        ^sub -> 1
        x when Integer.is_even(x) -> 2
        _ -> 4
      end)

    i = Nx.tensor(i_list, type: :f32)
    w = Nx.tensor(w_list, type: :f32)
    i_gpu = PolyHok.new_gnx(i)
    w_gpu = PolyHok.new_gnx(w)

    prev = System.monotonic_time()

    res =
      Ske.map(
        i_gpu,
        PolyHok.phok(fn i ->
          a = 0.0
          b = 3.1415926535
          sub = 2.0 * 1_048_575.0
          h = (b - a) / sub
          Ni.compute_xi(i, a, h)
        end)
      )
      |> Ske.map(PolyHok.phok(fn xi -> sinf(xi) end))
      |> Ske.map2(w_gpu, PolyHok.phok(fn fx, wi -> fx * wi end))
      |> Ske.reduce(0.0, PolyHok.phok(fn x, acc -> acc + x end))
      |> Ske.map(
        PolyHok.phok(fn s ->
          a = 0.0
          b = 3.1415926535
          sub = 2.0 * 1_048_575.0
          h = (b - a) / sub
          s * (h / 3.0)
        end)
      )
      |> PolyHok.get_gnx()

    next = System.monotonic_time()
    # IO.inspect(res)
    IO.puts("PolyHok\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
  end

  def run_fused() do
    sub = 2 * 1_048_575
    i_list = Enum.to_list(0..sub)

    w_list =
      Enum.map(i_list, fn
        0 -> 1
        ^sub -> 1
        x when Integer.is_even(x) -> 2
        _ -> 4
      end)

    i = Nx.tensor(i_list, type: :f32)
    w = Nx.tensor(w_list, type: :f32)
    i_gpu = PolyHok.new_gnx(i)
    w_gpu = PolyHok.new_gnx(w)

    prev = System.monotonic_time()

    res =
      Fusion.with_fusion(
        Ske.map(
          i_gpu,
          PolyHok.phok(fn i ->
            a = 0.0
            b = 3.1415926535
            sub = 2.0 * 1_048_575.0
            h = (b - a) / sub
            a + i * h
          end)
        )
        |> Ske.map(PolyHok.phok(fn xi -> sinf(xi) end))
        |> Ske.map2(w_gpu, PolyHok.phok(fn fx, wi -> fx * wi end))
        |> Ske.reduce(0.0, PolyHok.phok(fn x, acc -> acc + x end))
      )
      |> Ske.map(
        PolyHok.phok(fn s ->
          a = 0.0
          b = 3.1415926535
          sub = 2.0 * 1_048_575.0
          h = (b - a) / sub
          s * (h / 3.0)
        end)
      )
      |> PolyHok.get_gnx()
    next = System.monotonic_time()
    IO.inspect(res)
    IO.puts("PolyHok\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")

  end
end

Enum.each(1..10, fn x ->
  NiBenchmark.run_fused()
end)
