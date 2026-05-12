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
    :access_type -> :read, :write
    :rank -> dimensions do array [i,j,k] 
    :indices -> a tuple containing IndexExpr and RangeExpr
    :guards -> necessary conditions for pattern be valid, {:lt | :gte, :tid, :N}
    :ThreadMapping
    :precison -> :exact | :regular | :unknown  -> computed from lattice
    """

    defstruct [
      :array,
      :access_type,
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
    IO.inspect(a, label: "index a:")
    IO.inspect(b, label: "index b:")
    a == b
  end

  defp get_idx_expr(idx_expr) do
    %BoundAnalysis.IndexExpr{
      expr: strip_meta_from_access_indexes(idx_expr),
      range: nil
    }
  end

  defp build_dad({array_name, _meta, _meta_2}, access_pattern, access_type) do
    %BoundAnalysis.DAD{
      array: array_name,
      access_type: access_type,
      indices: get_idx_expr(access_pattern)
    }
  end

  defp add_access(summary, %BoundAnalysis.DAD{array: array, access_type: :read} = dad) do
    update_in(summary.reads, fn reads ->
      Map.update(reads, array, MapSet.new([dad]), fn dads ->
        MapSet.put(dads, dad)
      end)
    end)
  end

  defp add_access(summary, %BoundAnalysis.DAD{array: array, access_type: :write} = dad) do
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
              dad = build_dad(vector, access_pattern, access_type)
              acc = add_access(acc, dad)
              {access_node, acc}

            node ->
              {node, acc}
          end
      end)

    summary
  end

  defp traverse_kernel_ast(k_body, summary) do
    {_ast, summary} =
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
    traverse_kernel_ast(k_body, summary)
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

  defp build_fused_kernel(lhs_summary, rhs_summary, _dependency_list) do

    rename_map = lhs_rename_map(lhs_summary, rhs_summary)
    res = build_fused_params(lhs_summary, rhs_summary, rename_map)
    IO.inspect(res, label: "params")
    rx = lhs_summary.args ++ rhs_summary.args |> Enum.uniq()
    IO.inspect(rx, label: "args")

  end

  defmacro fuse(lhs, rhs) do
    sum1 = process_kernel(lhs)
    sum2 = process_kernel(rhs)
    r = raw_dependency_arrays(sum1, sum2)
    IO.inspect(r)

    if(not Enum.empty?(r)) do
      IO.puts("hehe")
      build_fused_kernel(sum1, sum2, r)
    end

    quote do
      :ok
    end
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
