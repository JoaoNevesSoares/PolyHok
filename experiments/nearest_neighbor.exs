require Integer
require PolyHok

defmodule DataSet do
  def open_data_set(file) do
    {:ok, contents} = File.read(file)

    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn f -> load_file(f) end)
    |> Enum.concat()
    |> Enum.concat()

    # |> Enum.unzip()
  end

  def load_file(file) do
    # IO.puts file
    {:ok, contents} = File.read(file)

    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn line ->
      words = String.split(line, " ", trim: true)
      [elem(Float.parse(Enum.at(words, 6)), 0), elem(Float.parse(Enum.at(words, 7)), 0)]
    end)
  end

  def gen_data_set_nx_double(n) do
    lat = 7 + Enum.random(0..63) + :rand.uniform()
    lon = Enum.random(0..358) + :rand.uniform()
    acc = <<lat::float-little-64, lon::float-little-64>>
    ref = gen_bin_data_double(n - 1, acc)

    %Nx.Tensor{
      data: %Nx.BinaryBackend{state: ref},
      type: {:f, 64},
      shape: {n, 2},
      names: [nil, nil]
    }
  end

  defp gen_bin_data_double(0, accumulator), do: accumulator

  defp gen_bin_data_double(size, accumulator) do
    lat = 7 + Enum.random(0..63) + :rand.uniform()
    lon = Enum.random(0..358) + :rand.uniform()

    gen_bin_data_double(
      size - 1,
      <<accumulator::binary, lat::float-little-64, lon::float-little-64>>
    )
  end

  def gen_data_set_nx(n) do
    {lat_ref, lng_ref} = gen_bin_data(n, <<>>, <<>>)

    {
      %Nx.Tensor{
        data: %Nx.BinaryBackend{state: lat_ref},
        type: {:f, 32},
        shape: {n},
        names: [nil]
      },
      %Nx.Tensor{
        data: %Nx.BinaryBackend{state: lng_ref},
        type: {:f, 32},
        shape: {n},
        names: [nil]
      }
    }
  end

  defp gen_bin_data(0, lat_accumulator, lng_accumulator), do: {lat_accumulator, lng_accumulator}

  defp gen_bin_data(size, lat_accumulator, lng_accumulator) do
    lat = 7 + Enum.random(0..63) + :rand.uniform()
    lon = Enum.random(0..358) + :rand.uniform()

    gen_bin_data(
      size - 1,
      <<lat_accumulator::binary, lat::float-little-32>>,
      <<lng_accumulator::binary, lon::float-little-32>>
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

defmodule HostNN do
  def distances(lat, lng) do
    lat
    |> Nx.multiply(lat)
    |> Nx.add(Nx.multiply(lng, lng))
    |> Nx.sqrt()
  end

  def stats(distances, lat, lng) do
    index =
      distances
      |> Nx.argmin()
      |> Nx.to_number()

    distances_list = Nx.to_flat_list(distances)
    lat_list = Nx.to_flat_list(lat)
    lng_list = Nx.to_flat_list(lng)
    min = Enum.at(distances_list, index)

    %{
      index: index,
      min: min,
      lat: Enum.at(lat_list, index),
      lng: Enum.at(lng_list, index),
      max: Enum.max(distances_list),
      first: Enum.take(distances_list, 8),
      last: distances_list |> Enum.reverse() |> Enum.take(8) |> Enum.reverse()
    }
  end

  def scalar(tensor) do
    tensor
    |> Nx.squeeze()
    |> Nx.to_number()
  end

  def comparison(gpu_value, host_value) do
    abs_error = abs(gpu_value - host_value)

    %{
      abs_error: abs_error,
      rel_error: abs_error / max(abs(host_value), 1.0e-12),
      matches_1e_5: abs_error <= 1.0e-5,
      matches_1e_4: abs_error <= 1.0e-4
    }
  end

  def buffer_comparison(gpu_distances, host_distances) do
    errors =
      gpu_distances
      |> Nx.to_flat_list()
      |> Enum.zip(Nx.to_flat_list(host_distances))
      |> Enum.map(fn {gpu, host} -> abs(gpu - host) end)

    %{
      max_abs_error: Enum.max(errors),
      mean_abs_error: Enum.sum(errors) / length(errors),
      matches_1e_5: Enum.all?(errors, &(&1 <= 1.0e-5)),
      matches_1e_4: Enum.all?(errors, &(&1 <= 1.0e-4))
    }
  end
end

PolyHok.defmodule NN do
  include(CAS_Float)
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

  defd euclid(lat, lng) do
    return sqrtf(lat * lat + lng * lng)
  end

  defd menor(x, y) do
    if x < y do
      x
    else
      y
    end
  end
end

use Ske

size = 8196

:rand.seed(:exsss, {123, 123, 123})

{lat_host, lng_host} = DataSet.gen_data_set_nx(size)
host_distances = HostNN.distances(lat_host, lng_host)
host_stats = HostNN.stats(host_distances, lat_host, lng_host)

lat_gpu = PolyHok.new_gnx(lat_host)
lng_gpu = PolyHok.new_gnx(lng_host)

prev = System.monotonic_time()

gpu_distances = Ske.map2(lat_gpu, lng_gpu, &NN.euclid/2)
gpu_res = Ske.reduce(gpu_distances, 50000.0, &NN.menor/2)

next = System.monotonic_time()

gpu_distances_host = PolyHok.get_gnx(gpu_distances)
res = PolyHok.get_gnx(gpu_res)
gpu_value = HostNN.scalar(res)
gpu_stats = HostNN.stats(gpu_distances_host, lat_host, lng_host)

IO.puts("PolyHok\t#{size}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
IO.inspect(lat_host, label: "host latitude buffer")
IO.inspect(lng_host, label: "host longitude buffer")
IO.inspect(host_distances, label: "host Nx distance buffer")
IO.inspect(host_stats, label: "host Nx nearest-neighbor stats")
IO.inspect(gpu_distances_host, label: "gpu Ske.map2 distance buffer")
IO.inspect(gpu_stats, label: "gpu Ske.map2 nearest-neighbor stats")

IO.inspect(HostNN.buffer_comparison(gpu_distances_host, host_distances),
  label: "gpu map2 buffer vs host Nx buffer comparison"
)

IO.inspect(res, label: "gpu Ske.reduce result buffer")

IO.inspect(HostNN.comparison(gpu_value, host_stats.min),
  label: "gpu reduce vs host Nx comparison"
)
