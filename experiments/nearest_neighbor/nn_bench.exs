Code.require_file("nn.exs", __DIR__)

defmodule Nnbench do
  def run(:fused, lat, lng, query_lat, query_lng) do
    start_time = System.monotonic_time()
    res = NN.run_fs(lat, lng, query_lat, query_lng)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def run(:unfused, lat, lng, query_lat, query_lng) do
    start_time = System.monotonic_time()
    res = NN.run_unfs(lat, lng, query_lat, query_lng)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def main() do
    argv = System.argv()

    {[{:file, in_file}, {:lat, lat_in}, {:lng, lng_in}, {:order, ord} | _rest], [], []} =
      OptionParser.parse(argv, strict: [file: :string, lat: :float, lng: :float, order: :integer])

    {lat, lg} =
      File.stream!(in_file)
      |> CSV.decode()
      |> Stream.map(fn {:ok, [lat, lg]} -> {String.to_float(lat), String.to_float(lg)} end)
      |> Enum.unzip()

    file = File.open!("results/nn/nn_#{in_file}_#{ord}.csv", [:write, :utf8])

    lat_gpu =
      Nx.tensor(lat)
      |> PolyHok.new_gnx()

    lng_gpu =
      Nx.tensor(lg)
      |> PolyHok.new_gnx()

    
    IO.inspect(lat_gpu, label: "lat gpu")
    IO.inspect(lng_gpu, label: "lng gpu")

    start = ord
    finish = 29 + ord

    res =
      Enum.reduce(start..finish, [], fn i, acc ->
        if rem(i, 2) == 0 do
          fs = run(:fused, lat_gpu, lng_gpu, lat_in, lng_in)
          ufs = run(:unfused, lat_gpu, lng_gpu, lat_in, lng_in)
          acc ++ [%{"unfused" => ufs, "fused" => fs}]
        else
          ufs = run(:unfused, lat_gpu, lng_gpu, lat_in, lng_in)
          fs = run(:fused, lat_gpu, lng_gpu, lat_in, lng_in)
          acc ++ [%{"unfused" => ufs, "fused" => fs}]
        end
      end)

    res
    |> CSV.encode(headers: ["unfused", "fused"])
    |> Enum.each(&IO.write(file, &1))

    File.close(file)
  end
end

Nnbench.main()
