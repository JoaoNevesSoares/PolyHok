defmodule Fusion do
  require PolyHok

  defp extract_ske_map({{:., _m1, [{:__aliases__, _m2, [:Ske]}, :map]}, _m3, args}) do
    {:ok, args}
  end

  defp extract_phok({{:., _m1, [{:__aliases__, _m2, [:PolyHok]}, :phok]}, _m3, [fun_ast]}) do
    {:ok, fun_ast}
  end

  defp peel_fn({:fn, _m, [{:->, _m2, [[param], body]}]}) do
    {param, body}
  end

  defp normalize_kernel_ops(ast) do
    case ast do
      # Kernel.+(...)
      {{:., meta, [{:__aliases__, _, [:Kernel]}, op]}, call_meta, args}
      when op in [:+, :-, :*, :/, :div, :rem] ->
        {op, Keyword.merge(call_meta, meta), Enum.map(args, &normalize_kernel_ops/1)}

      {form, meta, args} when is_list(args) ->
        {form, meta, Enum.map(args, &normalize_kernel_ops/1)}

      list when is_list(list) ->
        Enum.map(list, &normalize_kernel_ops/1)

      other ->
        other
    end
  end

  defp collect_param_names({n, _m, _ctx}) when is_atom(n), do: [n]

  defp collect_param_names({:=, _m, [lhs, _rhs]}), do: collect_param_names(lhs)

  defp collect_param_names({_, _m, list}) when is_list(list),
    do: Enum.flat_map(list, &collect_param_names/1)

  defp collect_param_names(list) when is_list(list),
    do: Enum.flat_map(list, &collect_param_names/1)

  defp collect_param_names(_), do: []

  defp put_meta({name, meta, ctx}, new_meta) when is_atom(name),
    do: {name, Keyword.merge(new_meta, meta), ctx}

  defp put_meta(other, _), do: other

  defp replace_in_clause({:->, m, [params, body]}, name, replacement) do
    param_names = params |> Enum.flat_map(&collect_param_names/1) |> MapSet.new()

    new_body =
      if MapSet.member?(param_names, name) do
        body
      else
        replace_var(body, name, replacement)
      end

    {:->, m, [params, new_body]}
  end

  defp replace_var(ast, name, replacement) when is_atom(name) do
    case ast do
      {:fn, m, clauses} ->
        {:fn, m, Enum.map(clauses, &replace_in_clause(&1, name, replacement))}

      {^name, meta, ctx} when is_atom(ctx) or is_nil(ctx) ->
        put_meta(replacement, meta)

      {form, meta, args} when is_list(args) ->
        {form, meta, Enum.map(args, &replace_var(&1, name, replacement))}

      list when is_list(list) ->
        Enum.map(list, &replace_var(&1, name, replacement))

      other ->
        other
    end
  end

  defp extract_device_name({:/, _, [inner, 1]}) do
    case inner do
      {{:., _, [_mod_ast, fun_name]}, _, _} when is_atom(fun_name) ->
        fun_name

      {fun_name, _, _} when is_atom(fun_name) ->
        fun_name
    end
  end

  defp build_fused_device_quote(data_ast, f_device_ast, [g_device_ast]) do
    f_name = extract_device_name(f_device_ast)
    g_name = extract_device_name(g_device_ast)

    # variáveis da DSL
    x_var = {:x, [], nil}
    y_var = {:y, [], nil}

    # type y int
    y_type_ast = {:y, [], [{:int, [], []}]}
    type_y = {:type, [], [y_type_ast]}

    # y = f(x)
    f_call = {f_name, [], [x_var]}
    assign_y = {:=, [], [y_var, f_call]}

    # return g(y)
    g_call = {g_name, [], [y_var]}
    return_stmt = {:return, [], [g_call]}

    body_block = {:__block__, [], [type_y, assign_y, return_stmt]}

    fused_fn_ast = {:fn, [], [{:->, [], [[x_var], body_block]}]}

    quote do
      Ske.map(
        unquote(data_ast),
        PolyHok.phok(unquote(fused_fn_ast))
      )
    end
  end

  defp build_fused_quote(data_ast, phok_f_wrapped, phok_g_wrapped, _caller_env) do
    with {:ok, f_ast} <- extract_phok(phok_f_wrapped),
         {:ok, g_ast} <- extract_phok(phok_g_wrapped) do
      {f_param, f_body} = peel_fn(f_ast)
      {g_param, g_body} = peel_fn(g_ast)

      f_name =
        case f_param do
          {name, _m, _ctx} when is_atom(name) -> name
          _ -> raise ArgumentError, "Parametro de f não reconhecido"
        end

      g_name =
        case g_param do
          {name, _m, _ctx} when is_atom(name) -> name
          _ -> raise ArgumentError, "Parametro de g não reconhecido"
        end

      # f_body continua em termos de f_name (ex: x + 1.0)
      # g_body(y) vira g_body(f_body(x)) substituindo y por f_body
      composed_body =
        g_body
        |> replace_var(g_name, f_body)
        # se você ainda estiver usando
        |> normalize_kernel_ops()

      # |> demote_int_floats()      # opcional, se precisar

      quote do
        Ske.map(
          unquote(data_ast),
          PolyHok.phok(fn unquote(f_param) ->
            unquote(composed_body)
          end)
        )
      end
    else
      _ ->
        raise ArgumentError, "with_fusion/1 requer PolyHok.phok(fn ... end)"
    end
  end

  defmacro with_fusion({:|>, _m, [lhs, rhs]}) do
    lhs_match = extract_ske_map(lhs)
    rhs_match = extract_ske_map(rhs)

    case {lhs_match, rhs_match} do
      {{:ok, [data_ast, {:&, _meta, [f_device]}]}, {:ok, [{:&, _, g_device}]}} ->
        build_fused_device_quote(data_ast, f_device, g_device)

      {{:ok, [data_ast, phok_f_ast]}, {:ok, [phok_g_ast]}} ->
        build_fused_quote(data_ast, phok_f_ast, phok_g_ast, __CALLER__)

      _ ->
        raise ArgumentError, "Erro na macro with_fusion"
    end
  end
end
