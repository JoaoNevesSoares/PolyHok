defmodule BoundAnalysis do
  require PolyHok

  defmodule IndexExpr do
    defstruct [
      :expr,
      :range
    ]
  end

  defmodule RangeExpr do
    defstruct [
      :expr,
      :range
    ]
  end

  defmodule ThreadMapping do
    defstruct [
      :tid,
      :block,
      :thread,
      :grid
    ]
  end

  defmodule DAD do
    @moduledoc false
    @doc """
    :array -> identificador do array
    :rank -> dimensions do array [i,j,k] 
    :indices -> a tuple containing IndexExpr and RangeExpr
    :guards -> necessary conditions for pattern be valid, {:lt | :gte, :tid, :N}
    :ThreadMapping
    :precison -> :exact | :regular | :unknown  -> computed from lattice
    """

    defstruct [
      :array,
      :indices
    ]
  end

  defmodule KernelSummary do
    defstruct [
      :name,
      :params,
      :args,
      reads: %{},
      writes: %{}
    ]
  end

  defp get_kernel_name({_header, _meta, [kernel_call, _t, _b, _args]}) do
    JIT.get_kernel_name(kernel_call)
  end

  defp strip_meta_from_access_indexes(ast) do
    Macro.prewalk(
      ast,
      fn
        {name, meta, args} when is_atom(name) and is_list(meta) ->
          {name, [], args}

        node ->
          node
      end
    )
  end

  defp equivalent_index?(a, b) do
    # IO.inspect(a, label: "index a:")
    # IO.inspect(b, label: "index b:")
    a == b
  end

  defp get_idx_expr(idx_expr) do
    %BoundAnalysis.IndexExpr{
      expr: strip_meta_from_access_indexes(idx_expr),
      range: nil
    }
  end

  defp build_dad({array_name, _meta, _meta_2}, access_pattern) do
    %BoundAnalysis.DAD{
      array: array_name,
      indices: get_idx_expr(access_pattern)
    }
  end

  defp add_access(:read, summary, %BoundAnalysis.DAD{array: array} = dad) do
    update_in(summary.reads, fn reads ->
      Map.update(reads, array, MapSet.new([dad]), fn dads ->
        MapSet.put(dads, dad)
      end)
    end)
  end

  defp add_access(:write, summary, %BoundAnalysis.DAD{array: array} = dad) do
    update_in(summary.writes, fn writes ->
      Map.update(writes, array, MapSet.new([dad]), fn dads ->
        MapSet.put(dads, dad)
      end)
    end)
  end

  defp collect_refs(summary, ast, access_type) do
    {_ast, summary} =
      Macro.prewalk(ast, summary, fn
        access_node, acc ->
          case access_node do
            {{:., meta_dot, [Access, :get]}, meta_access, [vector, access_pattern]}
            when is_list(meta_dot) and is_list(meta_access) ->
              dad = build_dad(vector, access_pattern)
              acc = add_access(access_type, acc, dad)
              {access_node, acc}

            node ->
              {node, acc}
          end
      end)

    summary
  end

  defp new_kernel_summary(k_name, k_params, k_args) do
    %BoundAnalysis.KernelSummary{
      name: k_name,
      params: k_params,
      args: k_args
    }
  end

  defp dataflow_analysis(summary, k_body) do
    {_ast, new_summary} =
      Macro.prewalk(k_body, summary, fn
        {:=, _meta, [lhs, rhs]} = node, acc ->
          acc =
            acc
            |> collect_refs(lhs, :write)
            |> collect_refs(rhs, :read)

          {node, acc}

        node, acc ->
          {node, acc}
      end)

    new_summary
  end

  defp get_kernel_args({_name, _meta, [_, _, _, args]}) do
    Macro.prewalk(args, fn node ->
      case node do
        {name, _meta, context} -> {name, [], context}
        _ -> node
      end
    end)
  end

  defp get_kernel_params({{:defk, _meta, [{_kernel_name, _call_meta, params}, _body]}, []}) do
    Enum.map(params, fn
      {name, _meta, context} -> {name, [], context}
    end)
  end

  defp process_kernel(ast) do
    name = get_kernel_name(ast)
    kernel_body = PolyHok.load_ast(name)
    kernel_args = get_kernel_args(ast)
    kernel_params = get_kernel_params(kernel_body)

    new_kernel_summary(name, kernel_params, kernel_args)
    |> dataflow_analysis(kernel_body)
  end

  defp var_name({name, _meta, nil}) when is_atom(name), do: name
  defp var_name(name) when is_atom(name), do: name

  defp param_arg_bindings(summary) do
    Enum.zip(summary.params, summary.args)
    |> Enum.map(fn {param, arg} ->
      {var_name(param), arg}
    end)
  end

  defp conflicting_params(lhs_summary, rhs_summary) do
    lhs = Map.new(param_arg_bindings(lhs_summary))
    rhs = Map.new(param_arg_bindings(rhs_summary))

    lhs
    |> Enum.filter(fn {param, lhs_arg} ->
      case Map.fetch(rhs, param) do
        {:ok, rhs_arg} -> lhs_arg != rhs_arg
        :error -> false
      end
    end)
    |> Enum.map(fn {param, _arg} -> param end)
  end

  defp lhs_rename_map(lhs_summary, rhs_summary) do
    lhs_summary
    |> conflicting_params(rhs_summary)
    |> Map.new(fn param ->
      {param, :"lhs_#{param}"}
    end)
  end

  defp build_fused_params(lhs_summary, rhs_summary, rename_map) do
    lhs_params =
      lhs_summary.params
      |> Enum.map(&var_name/1)
      |> Enum.map(fn param -> Map.get(rename_map, param, param) end)

    rhs_params =
      rhs_summary.params
      |> Enum.map(&var_name/1)

    (lhs_params ++ rhs_params) |> Enum.uniq()
  end

  def build_fused_args(lhs_summary, rhs_summary, rename_map) do
    # Essa função precisa saber a ordem dos parametros da kernel fusionado

    lhs =
      lhs_summary.params
      |> Enum.zip(lhs_summary.args)
      |> Enum.map(fn {param, arg} ->
        param_name = var_name(param)

        final_param = Map.get(rename_map, param_name, param_name)

        {final_param, arg}
      end)

    rhs =
      rhs_summary.params
      |> Enum.zip(rhs_summary.args)
      |> Enum.map(fn {param, arg} ->
        {var_name(param), arg}
      end)

    (lhs ++ rhs)
    |> Enum.uniq_by(fn {param, _arg} -> param end)
    |> Enum.map(fn {_param, arg} -> arg end)
  end

  defp rename_vars(ast, rename_map) do
    Macro.prewalk(ast, fn
      {name, meta, ctx} = node when is_atom(name) ->
        case Map.fetch(rename_map, name) do
          {:ok, new_name} -> {new_name, meta, ctx}
          :error -> node
        end

      node ->
        node
    end)
  end

  defp extract_defk_body({{:defk, _meta, [{_name, _call_meta, _params}, [do: body]]}, []}) do
    body
  end

  defp clean_var_context(ast) do
    Macro.prewalk(ast, fn
      {name, meta, ctx}
      when is_atom(name) and is_atom(ctx) ->
        {name, meta, nil}

      node ->
        node
    end)
  end

  defp build_spawn_ast(name, arity, args) do
    capture_ast =
      {:&, [],
       [
         {:/, [],
          [
            {{:., [], [{:__aliases__, [], [:SimpleTest]}, name]}, [no_parens: true], []},
            arity
          ]}
       ]}

    quote do
      PolyHok.spawn(
        unquote(capture_ast),
        {1, 1, 1},
        {10, 1, 1},
        [unquote_splicing(args)]
      )
    end
  end

  defp build_defk_ast(name, params, bodies) do
    params_ast =
      Enum.map(params, fn param ->
        {param, [], nil}
      end)

    statements =
      bodies
      |> Enum.flat_map(fn
        {:__block__, _meta, statements} -> statements
        statements -> [statements]
      end)

    body_ast =
      {:__block__, [], statements}

    {:defk, [],
     [
       {name, [], params_ast},
       [do: body_ast]
     ]}
  end

  defp same_array?({name, _meta, _ctx}, expected_name) when is_atom(name) do
    name == expected_name
  end

  defp same_array?(_, _), do: false

  defp strip_meta({op, _meta, args}) when is_atom(op) and is_list(args) do
    {op, [], Enum.map(args, &strip_meta/1)}
  end

  defp strip_meta({name, _meta, ctx}) when is_atom(name) and is_atom(ctx) do
    {name, [], nil}
  end

  defp strip_meta(list) when is_list(list) do
    Enum.map(list, &strip_meta/1)
  end

  defp strip_meta(other), do: other

  defp same_index?(left, right) do
    strip_meta(left) == strip_meta(right)
  end

  defp replace_reads(consumer_ast, array_name, index_expr, register_name) do
    Macro.postwalk(consumer_ast, fn
      {:=, meta, [lhs, rhs]} ->
        new_rhs =
          Macro.postwalk(rhs, fn
            {{:., _, [Access, :get]}, _, [array_ast, access_index]} = node ->
              if same_array?(array_ast, array_name) and same_index?(access_index, index_expr) do
                {register_name, [], nil}
              else
                node
              end

            node ->
              node
          end)

        {:=, meta, [lhs, new_rhs]}

      node ->
        node
    end)
  end

  defp replace_write(producer_ast, array_name, index_expr, register_name) do
    Macro.postwalk(producer_ast, fn
      {:=, meta, [lhs, rhs]} = node ->
        case lhs do
          {{:., _, [Access, :get]}, _, [array_ast, access_index]} ->
            if same_array?(array_ast, array_name) and same_index?(access_index, index_expr) do
              IO.inspect("Match!")
              {:=, meta, [{register_name, [], nil}, rhs]}
            else
              node
            end

          _ ->
            node
        end

      node ->
        node
    end)
  end

  defp fuse_register_forwarding(producer_ast, consumer_ast, dependency_info) do
    {producer_ast, consumer_ast, _counter} =
      Enum.reduce(dependency_info, {producer_ast, consumer_ast, 0}, fn {array_name, dad_set},
                                                                       {producer_ast,
                                                                        consumer_ast, counter} ->
        Enum.reduce(dad_set, {producer_ast, consumer_ast, counter}, fn dad,
                                                                       {producer_ast,
                                                                        consumer_ast, counter} ->
          register_name = :"fused__#{array_name}_#{counter}"

          producer_ast =
            replace_write(
              producer_ast,
              array_name,
              dad.indices.expr,
              register_name
            )

          consumer_ast =
            replace_reads(
              consumer_ast,
              array_name,
              dad.indices.expr,
              register_name
            )

          {producer_ast, consumer_ast, counter + 1}
        end)
      end)

    {producer_ast, consumer_ast}
  end

  defp build_fused_kernel(lhs_summary, rhs_summary, dependency_list) do
    IO.inspect(lhs_summary, label: "lhs summary struct")
    IO.inspect(rhs_summary, label: "rhs summary struct")
    rename_map = lhs_rename_map(lhs_summary, rhs_summary)

    fused_params = build_fused_params(lhs_summary, rhs_summary, rename_map)

    fused_args = build_fused_args(lhs_summary, rhs_summary, rename_map)

    lhs_body =
      lhs_summary.name
      |> PolyHok.load_ast()
      |> extract_defk_body()
      |> rename_vars(rename_map)
      |> clean_var_context()

    rhs_body =
      rhs_summary.name
      |> PolyHok.load_ast()
      |> extract_defk_body()
      |> clean_var_context()

    {new_lhs_body, new_rhs_body} = fuse_register_forwarding(lhs_body, rhs_body, dependency_list)

    fused_ast =
      build_defk_ast(:fused, fused_params, [new_lhs_body, new_rhs_body])
      |> clean_var_context()

    send(:module_server, {:add_ast, :fused, fused_ast, []})
    build_spawn_ast(:fused, length(fused_params), fused_args)
  end

  defmacro fuse(lhs, rhs) do
    sum1 = process_kernel(lhs)
    sum2 = process_kernel(rhs)

    interleaved_accesses = dependency_intersect(sum1.writes, sum2.reads)

    ast_fused =
      if(not Enum.empty?(interleaved_accesses)) do
        build_fused_kernel(sum1, sum2, interleaved_accesses)
      end

    ast_fused
  end

  defp dependency_intersect(producer_writes, consumer_reads) do
    Enum.reduce(producer_writes, %{}, fn {array_name, producer_dads}, acc ->
      consumer_dads = Map.get(consumer_reads, array_name, MapSet.new())
      matching_dads = MapSet.intersection(producer_dads, consumer_dads)

      if MapSet.size(matching_dads) > 0 do
        Map.put(acc, array_name, matching_dads)
      else
        acc
      end
    end)
  end

  defp raw_dependency_arrays(sum1, sum2) do
    sum1.writes
    |> Enum.flat_map(fn {array, write_dads} ->
      read_dads = Map.get(sum2.reads, array, MapSet.new())

      if dependent_dad_sets?(write_dads, read_dads) do
        [array]
      else
        []
      end
    end)
    |> Enum.uniq()
  end

  defp dependent_dad_sets?(write_dads, read_dads) do
    Enum.any?(write_dads, fn write_dad ->
      Enum.any?(read_dads, fn read_dad ->
        write_dad.array == read_dad.array and
          equivalent_index?(write_dad.indices.expr, read_dad.indices.expr)
      end)
    end)
  end
end
