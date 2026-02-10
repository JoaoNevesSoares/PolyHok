defmodule Fusion do
  defp fetch_defd!(map, name) do
    case Map.fetch(map, name) do
      {:ok, {defd_ast, _deps}} ->
        case defd_ast do
          {:defd, _meta, [_head, _body_kv]} ->
            defd_ast

          other ->
            raise ArgumentError,
                  "expected {:defd, _, [_, _]}, got: #{inspect(other, pretty: true)}"
        end
    end
  end

  defp build_combo_ast(f_atom, g_atom) do
    {:defd, [],
     [
       {:combo, [], [{:x, [], nil}]},
       [
         do:
           {:__block__, [],
            [
              {:type, [], [{:y, [], [{:int, [], nil}]}]},
              {:=, [],
               [
                 {:y, [], nil},
                 {f_atom, [], [{:x, [], nil}]}
               ]},
              {:return, [],
               [
                 {g_atom, [], [{:y, [], nil}]}
               ]}
            ]}
       ]
     ]}
  end

  @spec compose_devices(map(), atom(), atom()) :: {tuple(), [atom()]}
  def compose_devices(map, f_atom, g_atom) do
    _f = fetch_defd!(map, f_atom)
    _g = fetch_defd!(map, g_atom)

    ast = build_combo_ast(f_atom, g_atom)
    {ast, [f_atom, g_atom]}
  end
end

defmodule TestingJit do
  require PolyHok
  require PolyHokInspect
  require JIT
  use Ske

  def example() do
    PolyHok.defmodule PMap do
      defd inc(x) do
        x + 1
      end

      defd mult(y) do
        y * 2
      end

      defd combo2(x) do
        type(y(int))
        y = inc(x)
        return(mult(y))
      end
    end

    pid = self()
    f = :inc
    g = :mult
    send(:module_server, {:ast, pid})
    composed = receive do
      ast -> Fusion.compose_devices(ast, f, g)
    end
    mod_name = :Fused
    mod_body = {:__block__, [],
    [
      composed
    ]
    }
    JIT.process_module(mod_name, mod_body)
    send(:module_server, {:ast, pid})
    receive do
      ast -> IO.inspect(ast, label: "After Fusion")
    end
    n = 5
    arr1 = Nx.tensor(Enum.to_list(1..n), type: {:s, 32})
    host_res1 =
      arr1
      |> PolyHok.new_gnx()
      |> Ske.map(&Fused.combo/1)
      |> PolyHok.get_gnx()

    IO.inspect(host_res1)
  end
end

TestingJit.example()
