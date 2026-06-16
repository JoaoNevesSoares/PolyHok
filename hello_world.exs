require PolyHok
use Ske

PolyHok.defmodule Hello do
  # define a kernel, i.e., a function that will execute entirely on the GPU
  defk hello_kernel() do
    tid = threadIdx.x
    printf("helloworld")
  end

  # define a host entry point for computation
  def main() do
    threadsPerBlock = {10, 1, 1}
    numberOfBlocks = {1, 1, 1}

    h_arr = Nx.tensor(Enum.to_list(1..10), type: {:s, 32})
    d_arr = PolyHok.new_gnx(h_arr)

    PolyHok.spawn(
      &Hello.hello_kernel/0,
      numberOfBlocks,
      threadsPerBlock,
      []
    )

    PolyHok.get_gnx(d_arr)
  end

  def map_main() do

    PolyHok.random_gnx(1, 2, 100000000)
    |> Ske.map(PolyHok.phok(fn x -> x + 100 end))
    |> Ske.map(PolyHok.phok(fn x -> sqrtf(x) end))
    |> PolyHok.get_gnx()
    |> IO.inspect()
  end
end
Hello.map_main()
