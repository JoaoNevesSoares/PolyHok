
require PolyHok
require Fusion
use Ske

PolyHok.defmodule PMap do
  include(CAS)

  defd mult(xs, ys) do
    xs * ys
  end

  defd square(xs) do
    xs * xs
  end

  defd add(xs, ys) do
    xs + ys
  end

  defd f3(xs) do
    tanh(xs)
  end

  defd f4(xs) do
    xs / (1.0 + fabs(xs))
  end

  # # defd f5(xs) do
  # #   max(xs, 1.0)
  # end
end

random_generator = fn n, low, high ->
  key = Nx.Random.key(1234)
  {t, _key2} = Nx.Random.uniform(key, low, high, shape: {n}, type: {:f, 32})
  t
end

size_exp = 22 
size = Integer.pow(2, size_exp)
arr1 = random_generator.(size, -2.0, 3.0)

d_arr1 = PolyHok.new_gnx(arr1)

prev = System.monotonic_time()
host_res1 =
  Fusion.with_fusion(Ske.map(d_arr1,&PMap.square/1)
  |> Ske.map(&PMap.f3/1)
  |> Ske.map(&PMap.f4/1))
  |> PolyHok.get_gnx()
next = System.monotonic_time()

IO.puts(
  "PolyHok 2^#{size_exp}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}/ms"
)

IO.inspect(host_res1)
