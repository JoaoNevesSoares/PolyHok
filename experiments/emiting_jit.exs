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

  defmodule Emit do
    alias Fusion.Emit

    @spec emit_module({tuple(), [atom()]}, atom()) :: String.t()
    def emit_module({defd_ast, _deps}, module_name \\ :Fused) when is_atom(module_name) do
      defd_src = emit_defd(defd_ast)

      [
        "PolyHok.defmodule ",
        Atom.to_string(module_name),
        " do
",
        indent(defd_src, 2),
        "
end
"
      ]
      |> IO.iodata_to_binary()
    end

    @spec emit_defd(tuple()) :: String.t()
    def emit_defd({:defd, _m, [head, [do: block]]}) do
      {name, _m2, args} = head
      args_src = args |> Enum.map(&emit_expr/1) |> Enum.join(", ")
      body_src = emit_block(block)

      [
        "defd ",
        Atom.to_string(name),
        "(",
        args_src,
        ") do
",
        indent(body_src, 2),
        "
end"
      ]
      |> IO.iodata_to_binary()
    end

    @spec emit_block(tuple()) :: String.t()
    def emit_block({:__block__, _m, stmts}) when is_list(stmts) do
      stmts
      |> Enum.map(&emit_stmt/1)
      |> Enum.join("
")
    end

    defp emit_stmt({:type, _m, [{var, _mv, [type_ast]}]}) do
      ["type ", Atom.to_string(var), " ", emit_type(type_ast)] |> IO.iodata_to_binary()
    end

    defp emit_stmt({:=, _m, [{var, _mv, nil}, rhs]}) do
      [Atom.to_string(var), " = ", emit_expr(rhs)] |> IO.iodata_to_binary()
    end

    defp emit_stmt({:return, _m, [expr]}) do
      ["return ", emit_expr(expr)] |> IO.iodata_to_binary()
    end

    @spec emit_expr(term()) :: String.t()
    def emit_expr({var, _m, nil}) when is_atom(var), do: Atom.to_string(var)
    def emit_expr(int) when is_integer(int), do: Integer.to_string(int)
    def emit_expr(float) when is_float(float), do: :erlang.float_to_binary(float, [:compact])

    def emit_expr({fun, _m, args}) when is_atom(fun) and is_list(args) do
      inner = args |> Enum.map(&emit_expr/1) |> Enum.join(", ")
      [Atom.to_string(fun), "(", inner, ")"] |> IO.iodata_to_binary()
    end

    @spec emit_type(tuple()) :: String.t()
    def emit_type({name, _m, _}), do: Atom.to_string(name)

    # utils
    defp indent(str, n_spaces) when is_binary(str) do
      pad = String.duplicate(" ", n_spaces)

      str
      |> String.split("
")
      |> Enum.map_join(
        "
",
        fn line -> pad <> line end
      )
    end
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

    {ast, deps} =
      receive do
        ast -> Fusion.compose_devices(ast, f, g)
      end

    src = Fusion.Emit.emit_module({ast, deps}, :Ffused)
    IO.puts(src)

    n = 5
    arr1 = Nx.tensor(Enum.to_list(1..n), type: {:s, 32})
    host_res1 =
      arr1
      |> PolyHok.new_gnx()
      |> Ske.map(&Ffused.combo/1)
      |> PolyHok.get_gnx()
    IO.inspect(host_res1)
  end
end

TestingJit.example()
