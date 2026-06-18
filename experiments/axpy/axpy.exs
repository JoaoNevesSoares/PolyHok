require PolyHok
# require Fusion

n = 268_435_456

dev_a = PolyHok.random_gnx(0, 10, n)
dev_b = PolyHok.random_gnx(0, 10, n)

use Ske
a = 1.1
start_time = System.monotonic_time()
res = Ske.map2(dev_a, dev_b, PolyHok.clo(fn x, y -> a * x + y end))
end_time = System.monotonic_time()
res_cpu = PolyHok.get_gnx(res)
elapsed_ms = System.convert_time_unit(end_time - start_time, :native, :millisecond)
IO.inspect(res_cpu)
IO.puts("time #{elapsed_ms}")
