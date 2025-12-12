defmodule Mm do
  require PolyHok
  use Ske

  # helper de host: replica um escalar para o shape de outro tensor
  defp replicate_scalar(value, like_tensor) do
    value
    |> Nx.tensor()
    |> Nx.broadcast(Nx.shape(like_tensor))
  end

  def soft_max_enhanced() do
    # a = Nx.tensor([-1.3701, 0.7485, 0.1610, -2.0154, 1.0918])
    a =
  Nx.linspace(-1_000_000, 1_000_000, n: 2_000_000)
  |> Nx.to_tensor()

    gpu_a = PolyHok.new_gnx(a)

        # 1) max(x) no device
    gpu_max =
      Ske.reduce(
        gpu_a,
        -1.0e30,
        PolyHok.phok(fn acc, x ->
          if (x > acc) do
            return x
          end
          return acc
        end)
      )

    # 2) traz max para o host como número
    max_host =
      gpu_max
      |> PolyHok.get_gnx()
      |> Nx.squeeze()
      |> Nx.to_number()

    # 3) replica max com mesmo shape de 'a' e manda pra GPU
    max_vec = replicate_scalar(max_host, a)
    gpu_max_vec = PolyHok.new_gnx(max_vec)


    # Fusão 1 (denominador)
    gpu_deno = Ske.map2Reduce(gpu_a, gpu_max_vec,0.0,
    PolyHok.phok(fn (x, m) -> expf(x-m) end),
    PolyHok.phok(fn (acc, x) -> acc + x end))


    deno_host =
      gpu_deno
      |> PolyHok.get_gnx()
      |> Nx.squeeze()
      |> Nx.to_number()

    deno_vec = replicate_scalar(deno_host, a)
    gpu_deno_vec = PolyHok.new_gnx(deno_vec)

    gpu_softmax = Ske.map3(gpu_a, gpu_max_vec, gpu_deno_vec,
    PolyHok.phok(fn (x, m, deno) ->
      nominator = expf(x - m)
      return nominator / deno
    end))
    res = PolyHok.get_gnx(gpu_softmax)
    res= Nx.sum(res)
    IO.inspect(res, label: "softmax")
  end

  def soft_max() do
    #a = Nx.tensor([-1.3701, 0.7485, 0.1610, -2.0154, 1.0918])
        a =
  Nx.linspace(-1_000_000, 1_000_000, n: 2_000_000)
  |> Nx.to_tensor()
    gpu_a = PolyHok.new_gnx(a)

    # 1) max(x) no device
    gpu_max =
      Ske.reduce(
        gpu_a,
        -1.0e30,
        PolyHok.phok(fn acc, x ->
          if (x > acc) do
            return x
          end
          return acc
        end)
      )

    # 2) traz max para o host como número
    max_host =
      gpu_max
      |> PolyHok.get_gnx()
      |> Nx.squeeze()
      |> Nx.to_number()

    # 3) replica max com mesmo shape de 'a' e manda pra GPU
    max_vec = replicate_scalar(max_host, a)
    gpu_max_vec = PolyHok.new_gnx(max_vec)

    # 4) calcula exp(x - max) com map2, sem closure
    exp_shifted =
      Ske.map2(
        gpu_a,
        gpu_max_vec,
        PolyHok.phok(fn x, m ->
          expf(x - m)
        end)
      )

    # 5) soma dos exp(x - max) no device
    gpu_deno =
      Ske.reduce(
        exp_shifted,
        0.0,
        PolyHok.phok(fn (axx, x) -> axx + x end)
      )

    # 6) traz denominador para o host como número
    deno_host =
      gpu_deno
      |> PolyHok.get_gnx()
      |> Nx.squeeze()
      |> Nx.to_number()

    # 7) replica o denominador com mesmo shape de 'a' e manda pra GPU
    deno_vec = replicate_scalar(deno_host, a)
    gpu_deno_vec = PolyHok.new_gnx(deno_vec)

    # 8) normaliza: num / deno elemento a elemento com map2
    res =
      Ske.map2(
        exp_shifted,
        gpu_deno_vec,
        PolyHok.phok(fn num, deno ->
          num / deno
        end)
      )
      |> PolyHok.get_gnx()

    res = Nx.sum(res)
    IO.inspect(res, label: "softmax")
  end
end

prev = System.monotonic_time()
Mm.soft_max_enhanced()
next = System.monotonic_time()
IO.puts "PolyHok\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

#Mm.soft_max()
#Mm.soft_max_enhanced()
