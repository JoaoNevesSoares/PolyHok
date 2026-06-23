require PolyHok
use Ske
require Fusion

PolyHok.defmodule NN do
  defd candidate_distance(lat_i, lng_i, target_lat, target_lng) do
    dx = target_lat - lat_i
    dy = target_lng - lng_i
    dist = sqrtf(dx * dx + dy * dy)
    return(dist)
  end

  defd min_candidate(d1, d2) do
    if d1 < d2 do
      d1
    else
      d2
    end
  end

  def run_unfs(lat, lng, target_lat, target_lng) do
    res =
      Ske.map2(
        lat,
        lng,
        PolyHok.clo(fn lat_i, lng_i ->
          NN.candidate_distance(lat_i, lng_i, target_lat, target_lng)
        end)
      )
      |> Ske.reduce(
        100000000.0, &NN.min_candidate/2)
      res
  end

  def run_fs(lat, lng, target_lat, target_lng) do
    res =
    Fusion.with_fusion(
      Ske.map2(
        lat,
        lng,
        PolyHok.clo(fn lat_i, lng_i ->
          NN.candidate_distance(lat_i, lng_i, target_lat, target_lng)
        end)
      )
      |> Ske.reduce(
        100000000.0, &NN.min_candidate/2))
      res
  end
  # def main() do
  #   {lat, lg} =
  #     File.stream!("cane04.csv")
  #     |> CSV.decode()
  #     |> Stream.map(fn {:ok, [lat, lg]} -> {String.to_float(lat), String.to_float(lg)} end)
  #     |> Enum.unzip()
  #
  #   lat_tensor = Nx.tensor(lat)
  #   lg_tensor = Nx.tensor(lg)
  #   lat_gpu = PolyHok.new_gnx(lat_tensor)
  #   lng_gpu = PolyHok.new_gnx(lg_tensor)
  #   run_fs(lat_gpu, lng_gpu, 25.0, 290.0)
  #   |> PolyHok.get_gnx()
  #   |> IO.inspect(label: "result")
  # end
end

# NN.main()
