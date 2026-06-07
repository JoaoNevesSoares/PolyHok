defmodule FusionCacheTest do
  use ExUnit.Case, async: false

  require Fusion

  setup do
    stop_module_server()

    pid = spawn_link(fn -> JIT.module_server(%{}, %{}) end)
    Process.register(pid, :module_server)

    on_exit(fn -> stop_module_server() end)

    :ok
  end

  test "same map chain reuses one cached fused function" do
    first =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, PolyHok.phok(fn x -> x + 1 end))
            |> Ske.map(PolyHok.phok(fn y -> y * 2 end))
          )
        end
      )

    assert [name] = anon_names(first)
    assert fusion_cache_size() == 1

    second =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, PolyHok.phok(fn x -> x + 1 end))
            |> Ske.map(PolyHok.phok(fn y -> y * 2 end))
          )
        end
      )

    assert [^name] = anon_names(second)
    assert fusion_cache_size() == 1
  end

  test "runtime tensor names are not part of the cache identity" do
    first =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, PolyHok.phok(fn x -> x + 1 end))
            |> Ske.map(PolyHok.phok(fn y -> y * 2 end))
          )
        end
      )

    second =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(other_tensor, PolyHok.phok(fn x -> x + 1 end))
            |> Ske.map(PolyHok.phok(fn y -> y * 2 end))
          )
        end
      )

    assert anon_names(first) == anon_names(second)
    assert fusion_cache_size() == 1
  end

  test "kernel order changes the fused function identity" do
    first =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, PolyHok.phok(fn x -> x + 1 end))
            |> Ske.map(PolyHok.phok(fn y -> y * 2 end))
          )
        end
      )

    second =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, PolyHok.phok(fn x -> x * 2 end))
            |> Ske.map(PolyHok.phok(fn y -> y + 1 end))
          )
        end
      )

    assert anon_names(first) != anon_names(second)
    assert fusion_cache_size() == 2
  end

  test "folded scalar literals are part of the cache identity" do
    two =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map2(a, 2, PolyHok.phok(fn x, y -> x + y end))
            |> Ske.map(PolyHok.phok(fn z -> z * 2 end))
          )
        end
      )

    three =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map2(a, 3, PolyHok.phok(fn x, y -> x + y end))
            |> Ske.map(PolyHok.phok(fn z -> z * 2 end))
          )
        end
      )

    assert anon_names(two) != anon_names(three)
    assert fusion_cache_size() == 2
  end

  test "dynamic input aliasing is preserved in the cache identity" do
    same_input =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map2(a, a, PolyHok.phok(fn x, y -> x + y end))
            |> Ske.map(PolyHok.phok(fn z -> z * 2 end))
          )
        end
      )

    different_inputs =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map2(a, b, PolyHok.phok(fn x, y -> x + y end))
            |> Ske.map(PolyHok.phok(fn z -> z * 2 end))
          )
        end
      )

    assert anon_names(same_input) != anon_names(different_inputs)
    assert fusion_cache_size() == 2
  end

  test "device function with single if body can be fused" do
    register_device_function(
      :inv,
      quote do
        defd inv(t) do
          if t != 0.0 do
            t * 1.0
          else
            t + 1.0
          end
        end
      end
    )

    expanded =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, PolyHok.phok(fn x -> x * 3.0 end))
            |> Ske.map(&Dp.inv/1)
          )
        end
      )

    assert [_name] = anon_names(expanded)
    assert fusion_cache_size() == 1
  end

  test "single skeleton input is returned unchanged" do
    ast =
      quote do
        Fusion.with_fusion(Ske.reduce(a, 0, PolyHok.phok(fn x, acc -> x + acc end)))
      end

    expanded = expand_fusion(ast)

    assert Macro.to_string(expanded) ==
             Macro.to_string(quote(do: Ske.reduce(a, 0, PolyHok.phok(fn x, acc -> x + acc end))))
  end

  test "valid map3 pipe shape is accepted" do
    expanded =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, PolyHok.phok(fn x -> x end))
            |> Ske.map3(b, c, PolyHok.phok(fn x, y, z -> x + y + z end))
          )
        end
      )

    assert [_name] = anon_names(expanded)
  end

  test "valid map4 pipe shape is accepted" do
    expanded =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, PolyHok.phok(fn x -> x end))
            |> Ske.map4(b, c, d, PolyHok.phok(fn x, y, z, w -> x + y + z + w end))
          )
        end
      )

    assert [_name] = anon_names(expanded)
  end

  test "four input map chain emits map4" do
    expanded =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map4(a, b, c, d, PolyHok.phok(fn w, x, y, z -> w + x + y + z end))
            |> Ske.map(PolyHok.phok(fn total -> total * 2 end))
          )
        end
      )

    expanded_string = Macro.to_string(expanded)

    assert expanded_string =~ "Ske.map4("
    assert [_name] = anon_names(expanded)
  end

  test "four input map chain with terminal reduce emits map4Reduce" do
    expanded =
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map4(a, b, c, d, PolyHok.phok(fn w, x, y, z -> w + x + y + z end))
            |> Ske.reduce(0, PolyHok.phok(fn value, acc -> value + acc end))
          )
        end
      )

    expanded_string = Macro.to_string(expanded)

    assert expanded_string =~ "Ske.map4Reduce("
    assert [_name] = anon_names(expanded)
  end

  test "single non-skeleton input is returned unchanged" do
    expanded =
      expand_fusion(
        quote do
          Fusion.with_fusion(foo(a))
        end
      )

    assert Macro.to_string(expanded) == Macro.to_string(quote(do: foo(a)))
  end

  test "malformed skeleton arity in a pipe raises clear fusion error" do
    assert_raise ArgumentError, ~r/malformed Ske\.map call/, fn ->
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, b, PolyHok.phok(fn x -> x end))
            |> Ske.map(PolyHok.phok(fn x -> x end))
          )
        end
      )
    end
  end

  test "malformed map4 arity in a pipe raises clear fusion error" do
    assert_raise ArgumentError, ~r/malformed Ske\.map4 call/, fn ->
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map4(a, b, PolyHok.phok(fn x, y, z, w -> x + y + z + w end))
            |> Ske.map(PolyHok.phok(fn x -> x end))
          )
        end
      )
    end
  end

  test "five input map fusion raises clear arity limit error" do
    assert_raise ArgumentError, ~r/full-chain fusion requires <= 4 tensor inputs/, fn ->
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map4(a, b, c, d, PolyHok.phok(fn w, x, y, z -> w + x + y + z end))
            |> Ske.map2(e, PolyHok.phok(fn total, extra -> total + extra end))
          )
        end
      )
    end
  end

  test "unsupported Ske skeleton in a pipe raises clear fusion error" do
    assert_raise ArgumentError, ~r/does not support Ske\.scan/, fn ->
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.scan(a, PolyHok.phok(fn x -> x end))
            |> Ske.map(PolyHok.phok(fn x -> x end))
          )
        end
      )
    end
  end

  test "local calls in a pipe remain rejected" do
    assert_raise ArgumentError, ~r/expects explicit Ske\.map/, fn ->
      expand_fusion(
        quote do
          Fusion.with_fusion(
            map(a, PolyHok.phok(fn x -> x end))
            |> Ske.map(PolyHok.phok(fn x -> x end))
          )
        end
      )
    end
  end

  test "piped call shape is rejected as first stage in a pipe" do
    assert_raise ArgumentError, ~r/first stage map expects standalone map shape/, fn ->
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(PolyHok.phok(fn x -> x end))
            |> Ske.map(PolyHok.phok(fn x -> x end))
          )
        end
      )
    end
  end

  test "standalone call shape is rejected after a pipe" do
    assert_raise ArgumentError, ~r/stage 2 map expects piped map shape/, fn ->
      expand_fusion(
        quote do
          Fusion.with_fusion(
            Ske.map(a, PolyHok.phok(fn x -> x end))
            |> Ske.map(b, PolyHok.phok(fn x -> x end))
          )
        end
      )
    end
  end

  defp expand_fusion(ast), do: Macro.expand(ast, __ENV__)

  defp register_device_function(name, {:defd, _meta, [header, body]} = ast) do
    funs = JIT.find_functions(ast)
    send(:module_server, {:add_ast, name, {:defd, [], [header, body]}, funs})
  end

  defp anon_names(ast) do
    {_ast, names} =
      Macro.prewalk(ast, [], fn
        {:{}, _, [:anon, name, _]} = node, names when is_binary(name) ->
          {node, [name | names]}

        node, names ->
          {node, names}
      end)

    Enum.reverse(names)
  end

  defp fusion_cache_size do
    send(:module_server, {:fusion_map, self()})

    receive do
      {:fusion_map, map} -> map_size(map)
    end
  end

  defp stop_module_server do
    case Process.whereis(:module_server) do
      nil ->
        :ok

      pid ->
        try do
          Process.unregister(:module_server)
        rescue
          ArgumentError -> :ok
        end

        if Process.alive?(pid) do
          send(pid, {:kill})
        end
    end
  end
end
