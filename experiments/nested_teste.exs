require PolyHok
use Ske

h_arr = Nx.tensor(Enum.to_list(1..10), type: {:f, 32})
d_arr = PolyHok.new_gnx(h_arr)

res = Ske.map(d_arr, PolyHok.phok fn x -> x * x end)
h_res = PolyHok.get_gnx(res)

IO.inspect(h_res)
