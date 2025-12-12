defmodule Fusion do
  require PolyHok

  defmodule AstCall do
    @moduledoc false

    @doc """
     :ske        -> :map, :reduce, :scan...
     :data_ast   -> input arrays for skeletons and scalars
     :kernel_ast -> raw AST
    """
    defstruct [:ske, :data_ast, :kernel_ast]
  end

  defp new_skecall(ske_name, data_ast, kernel_ast) do
    %AstCall{
      ske: ske_name,
      data_ast: data_ast,
      kernel_ast: kernel_ast
    }
  end

  defp parse_ske_call({{:., _meta1, [{_alias, _meta2, [:Ske]}, ske_name]}, _meta3, args})
       when ske_name in [:map, :map2, :reduce] do
    do_parse_ske_call(ske_name, args)
  end

  defp do_parse_ske_call(ske_name, args) do
    {data_ast, kernel_ast} =
      case {ske_name, args} do
        {:map, [kernel_ast]} -> {nil, kernel_ast}
        {:map, [data_ast, kernel_ast]} -> {data_ast, kernel_ast}
        {:map2, [data_ast, kernel_ast]} -> {data_ast, kernel_ast}
        {:map2, [data1, data2, kernel_ast]} -> {[data1, data2], kernel_ast}
        {:reduce, [data_ast, kernel_ast]} -> {data_ast, kernel_ast}
      end

    new_skecall(ske_name, data_ast, kernel_ast)
  end

  defp split_body_and_return(body) do
    body = List.wrap(body)

    case body do
      [] ->
        raise ArgumentError, "Empty function body in split_body_and_return/1"

      _ ->
        {prefix, [last]} = Enum.split(body, length(body) - 1)

        case last do
          {:return, _meta, [expr]} ->
            {prefix, expr}

          other ->
            {prefix, other}
        end
    end
  end

  defp merge_two_function(f_args, f_body, [x_arg], g_body) do
    f_body = List.wrap(f_body)
    g_body = List.wrap(g_body)

    {x_var, x_meta, x_ctx} = x_arg
    tmp_x_var = :"tmp_#{x_var}"
    tmp_x_ast = {tmp_x_var, x_meta, x_ctx}
    {f_body_prefix, f_ret_expr} = split_body_and_return(f_body)

    rename_vars_g = fn ast ->
      Macro.prewalk(ast, fn
        {var, meta, ctx} = node
        when is_atom(var) and (is_atom(ctx) or is_nil(ctx)) ->
          cond do
            var in [x_var, tmp_x_var, :return, :type] ->
              node

            true ->
              {:"tmp_#{var}", meta, ctx}
          end

        other ->
          other
      end)
    end

    g_body1 = rename_vars_g.(g_body)

    g_body2 =
      Macro.prewalk(g_body1, fn
        {^x_var, meta, ctx} -> {tmp_x_var, meta, ctx}
        other -> other
      end)

    tmp_x_binding = {:=, x_meta, [tmp_x_ast, f_ret_expr]}
    f_body_prefix ++ [tmp_x_binding | g_body2]
  end

  defp build_map_reduce_call(
         %AstCall{ske: :map} = lhs,
         %AstCall{ske: :reduce} = rhs
       ) do
    map_f = normalize_kernel_ast(lhs.kernel_ast)
    red_f = normalize_kernel_ast(rhs.kernel_ast)

    quote do
      Ske.mapReduce(
        unquote(lhs.data_ast),
        unquote(rhs.data_ast),
        unquote(map_f),
        unquote(red_f)
      )
    end
  end

  defp build_map_reduce_call(
         %AstCall{ske: :map2, data_ast: [t1, t2]} = lhs,
         %AstCall{ske: :reduce} = rhs
       ) do
    map_f = normalize_kernel_ast(lhs.kernel_ast)
    red_f = normalize_kernel_ast(rhs.kernel_ast)

    quote do
      Ske.map2Reduce(
        unquote(t1),
        unquote(t2),
        unquote(rhs.data_ast),
        unquote(map_f),
        unquote(red_f)
      )
    end
  end

  defp build_phok_fun(args, body) do
    quote do
      PolyHok.phok(fn unquote_splicing(args) ->
        (unquote_splicing(body))
      end)
    end
  end

  defp decompose_kernel(kernel_ast) do
    case kernel_ast do
      # PolyHok.phok(...)
      {{:., _, [{:__aliases__, _, [:PolyHok]}, :phok]}, _, _} ->
        comp_ast_phok(kernel_ast)

      # &Mod.fun/arity
      {:&, _, _} ->
        comp_ast_device(kernel_ast)

      other ->
        raise ArgumentError,
              "unsupported kernel AST in Fusion: #{Macro.to_string(other)}"
    end
  end

  defp comp_ast_phok(
         {{:., _, [{:__aliases__, _, [:PolyHok]}, :phok]}, _,
          [{:fn, _, [{:->, _, [args, body_ast]}]}]}
       ) do
    {args, normalize_body_ast(body_ast)}
  end

  defp comp_ast_device(
         {:&, _, [{:/, _, [{{:., _, [{:__aliases__, _, _}, f_name]}, _, []}, f_arity]}]}
       ) do
    pid = self()
    send(:module_server, {:get_ast, f_name, pid})

    {{:defd, _m, fn_body}, _} =
      receive do
        {:ast, body} -> body
      end

    [args | block] = fn_body
    args = extract_args(args)
    block = extract_block(block)
    {args, block}
  end

  defp extract_args({_name, _m, args}) do
    args
  end

  defp extract_block([[do: body_ast]]), do: normalize_body_ast(body_ast)
  defp extract_block(do: body_ast), do: normalize_body_ast(body_ast)

  defp normalize_body_ast(body_ast) do
    case body_ast do
      {:__block__, _m, block_body} when is_list(block_body) ->
        case block_body do
          [{:return, _rm, [expr]}] -> [expr]
          other -> other
        end

      {:return, _m, [expr]} ->
        [expr]

      expr ->
        [expr]
    end
  end

  defp normalize_kernel_ast(kernel_ast) do
    {args, body} = decompose_kernel(kernel_ast)
    build_phok_fun(args, body)
  end

  defmacro with_fusion({:|>, _m, [lhs, rhs]}) do
    lhs_call = parse_ske_call(lhs)
    rhs_call = parse_ske_call(rhs)
    apply_fusion(lhs_call, rhs_call)
  end

  defp apply_fusion(
         %AstCall{ske: :map} = lhs,
         %AstCall{ske: :map} = rhs
       ) do
    {f_args, f_body} = decompose_kernel(lhs.kernel_ast)
    {g_args, g_body} = decompose_kernel(rhs.kernel_ast)

    fused_body = merge_two_function(f_args, f_body, g_args, g_body)
    fun = build_phok_fun(f_args, fused_body)

    quote do
      Ske.map(unquote(lhs.data_ast), unquote(fun))
    end
  end

  defp apply_fusion(
         %AstCall{ske: ske} = lhs,
         %AstCall{ske: :reduce} = rhs
       )
       when ske in [:map, :map2] do
    build_map_reduce_call(lhs, rhs)
  end

  defp apply_fusion(
         %AstCall{ske: :map2} = lhs,
         %AstCall{ske: :map2} = rhs
       ) do
    {f_args, f_body} = decompose_kernel(lhs.kernel_ast)
    {g_args, g_body} = decompose_kernel(rhs.kernel_ast)

    [c_arg, d_arg] = g_args
    {d_atom, m1, m2} = d_arg
    h_args = f_args ++ [{:"tmp_#{d_atom}", m1, m2}]
    fused_body = merge_two_function(f_args, f_body, [c_arg], g_body)

    fun = build_phok_fun(h_args, fused_body)
    [t1, t2] = lhs.data_ast
    t3 = rhs.data_ast

    quote do
      Ske.map3(unquote(t1), unquote(t2), unquote(t3), unquote(fun))
    end
  end
end
