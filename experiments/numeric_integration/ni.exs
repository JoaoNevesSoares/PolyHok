require PolyHok
require Integer
use Ske

PolyHok.defmodule Ni do
  defd compute_xi(i, a, h) do
    a + i * h
  end

  def ni_unfs() do
    n = 100.0
    intervals = 2.0 * n
    m = intervals + 1.0
    a = 0.0
    b = 3.14159265
    h = (b - a) / intervals

    i_list = Enum.to_list(0..(round(m) - 1))

    w_list =
      Enum.map(i_list, fn
        x when x == 0 -> 1
        x when x == 2 * round(n) -> 1
        x when Integer.is_even(x) -> 2
        x when Integer.is_odd(x) -> 4
      end)

    i = Nx.tensor(i_list, type: :f32)
    w = Nx.tensor(w_list, type: :f32)
    i_gpu = PolyHok.new_gnx(i)
    w_gpu = PolyHok.new_gnx(w)

    sum =
      Ske.map(
        i_gpu,
        PolyHok.clo(fn i ->
          Ni.compute_xi(i, a, h)
        end)
      )
      |> Ske.map(
        PolyHok.phok(fn xi ->
          sinf(xi)
        end)
      )
      |> Ske.map2(
        w_gpu,
        PolyHok.phok(fn fx, wi ->
          fx * wi
        end)
      )
      |> Ske.reduce(0.0, PolyHok.phok(fn x, acc -> acc + x end))

    host_sum = PolyHok.get_gnx(sum)
    compute = Nx.to_number(host_sum[0][0])
    h = 0.0157079633
    result = compute * h / 3.0
    IO.inspect(result)
  end
end

Ni.ni_unfs()
