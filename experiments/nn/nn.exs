require Integer
require PolyHok

defmodule DataSet do
  def gen_data_set_nx_double(n) do
    {x_ref, y_ref} = gen_bin_data_double(n, <<>>, <<>>)

    {
      %Nx.Tensor{data: %Nx.BinaryBackend{state: x_ref}, type: {:f, 64}, shape: {n}, names: [nil]},
      %Nx.Tensor{data: %Nx.BinaryBackend{state: y_ref}, type: {:f, 64}, shape: {n}, names: [nil]}
    }
  end

  defp gen_bin_data_double(0, x_accumulator, y_accumulator), do: {x_accumulator, y_accumulator}

  defp gen_bin_data_double(size, x_accumulator, y_accumulator) do
    x = 7 + Enum.random(0..63) + :rand.uniform()
    y = Enum.random(0..358) + :rand.uniform()

    gen_bin_data_double(
      size - 1,
      <<x_accumulator::binary, x::float-little-64>>,
      <<y_accumulator::binary, y::float-little-64>>
    )
  end

  def gen_data_set_nx(n) do
    {x_ref, y_ref} = gen_bin_data(n, <<>>, <<>>)

    {
      %Nx.Tensor{data: %Nx.BinaryBackend{state: x_ref}, type: {:f, 32}, shape: {n}, names: [nil]},
      %Nx.Tensor{data: %Nx.BinaryBackend{state: y_ref}, type: {:f, 32}, shape: {n}, names: [nil]}
    }
  end

  defp gen_bin_data(0, x_accumulator, y_accumulator), do: {x_accumulator, y_accumulator}

  defp gen_bin_data(size, x_accumulator, y_accumulator) do
    x = 7 + Enum.random(0..63) + :rand.uniform()
    y = Enum.random(0..358) + :rand.uniform()

    gen_bin_data(
      size - 1,
      <<x_accumulator::binary, x::float-little-32>>,
      <<y_accumulator::binary, y::float-little-32>>
    )
  end

  def gen_data_set(n), do: gen_data_set_(n, [])
  def gen_data_set_(0, data), do: data

  def gen_data_set_(n, data) do
    lat = 7 + Enum.random(0..63) + :rand.uniform()
    lon = Enum.random(0..358) + :rand.uniform()
    gen_data_set_(n - 1, [lat, lon | data])
  end

  def gen_lat_long(_l, c) do
    if(Integer.is_even(c)) do
      Enum.random(0..358) + :rand.uniform()
    else
      7 + Enum.random(0..63) + :rand.uniform()
    end
  end
end

PolyHok.defmodule NN do
  require Fusion
  def euclid_seq(l, lat, lng), do: euclid_seq_(l, lat, lng, [])

  def euclid_seq_([m_lat, m_lng | array], lat, lng, data) do
    # m_lat = Enum.at(array,0)
    # m_lng = Enum.at(array,1)

    value = :math.sqrt((lat - m_lat) * (lat - m_lat) + (lng - m_lng) * (lng - m_lng))
    # value = :math.sqrt((lat-m_lat)*(lat-m_lat)+(lng-m_lng)*(lng-m_lng))
    euclid_seq_(array, lat, lng, [value | data])
  end

  def euclid_seq_([], _lat, _lng, data) do
    data
  end

  defd euclid(x, y) do
    return(sqrtf(x * x + y * y))
  end

  defd menor(x, y) do
    if x < y do
      x
    else
      y
    end
  end

  def run_fused() do
    size = 400_000_000 
    {x_host, y_host} = DataSet.gen_data_set_nx(size)
    prev = System.monotonic_time()
    x_gpu = PolyHok.new_gnx(x_host)
    y_gpu = PolyHok.new_gnx(y_host)
    r =
      Fusion.with_fusion(
      Ske.map2(x_gpu, y_gpu, &NN.euclid/2)
      |> Ske.reduce(50000.0, &NN.menor/2))
      |> PolyHok.get_gnx()

    next = System.monotonic_time()
    PolyHok.get_gnx(x_gpu)
    PolyHok.get_gnx(y_gpu)
    IO.puts("PolyHok\t#{size}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
  end

  def run_unfused() do
    size =  400_000_000
    {x_host, y_host} = DataSet.gen_data_set_nx(size)
    prev = System.monotonic_time()
    x_gpu = PolyHok.new_gnx(x_host)
    y_gpu = PolyHok.new_gnx(y_host)
    r =
      Ske.map2(x_gpu, y_gpu, &NN.euclid/2)
      |> Ske.reduce(50000.0, &NN.menor/2)
      |> PolyHok.get_gnx()

    next = System.monotonic_time()
    PolyHok.get_gnx(x_gpu)
    PolyHok.get_gnx(y_gpu)
    IO.puts("PolyHok\t#{size}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
  end
end

use Ske

Enum.each(1..10, fn x -> 
  # NN.run_unfused()
  NN.run_fused()
end)
