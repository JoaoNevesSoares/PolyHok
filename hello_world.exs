require PolyHok
use Ske

####
#
# this is an example on how to use the erlang nif to generate random data directly
# into GPU global memory
#
####

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
    PolyHok.random_gnx(5, 10, 256, {:s, 32})
    |> PolyHok.get_gnx()
    |> IO.inspect(label: "integer random")

    PolyHok.random_gnx(5, 10, 256, {:f, 32})
    |> PolyHok.get_gnx()
    |> IO.inspect(label: "float")

    PolyHok.random_gnx(5, 10, 256, {:f, 64})
    |> PolyHok.get_gnx()
    |> IO.inspect(label: "double")
  end
end
Hello.map_main()
