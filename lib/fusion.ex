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
       when ske_name in [:map, :map2, :map3, :reduce] do
    do_parse_ske_call(ske_name, args)
  end

  defp parse_ske_call(other) do
    raise ArgumentError,
          "Fusion.with_fusion/1 expects Ske.map/map2/map3/reduce calls in a pipe chain, got: #{Macro.to_string(other)}"
  end

  defp do_parse_ske_call(ske_name, args) do
    {data_ast, kernel_ast} =
      case {ske_name, args} do
        {:map, [kernel_ast]} -> {nil, kernel_ast}
        {:map, [data_ast, kernel_ast]} -> {data_ast, kernel_ast}
        {:map2, [data_ast, kernel_ast]} -> {data_ast, kernel_ast}
        {:map2, [data1, data2, kernel_ast]} -> {[data1, data2], kernel_ast}
        {:map3, [data1, data2, kernel_ast]} -> {[data1, data2], kernel_ast}
        {:map3, [data1, data2, data3, kernel_ast]} -> {[data1, data2, data3], kernel_ast}
        {:reduce, [data_ast, kernel_ast]} -> {data_ast, kernel_ast}
        {:reduce, [data_ast, initial, kernel_ast]} -> {[data_ast, initial], kernel_ast}
      end

    new_skecall(ske_name, data_ast, kernel_ast)
  end

  defp flatten_pipe_ast({:|>, _meta, [lhs, rhs]}) do
    flatten_pipe_ast(lhs) ++ flatten_pipe_ast(rhs)
  end

  defp flatten_pipe_ast(ast), do: [ast]

  defp stage_arity(:map), do: 1
  defp stage_arity(:map2), do: 2
  defp stage_arity(:map3), do: 3

  defp normalize_data_list(nil), do: []
  defp normalize_data_list(data) when is_list(data), do: data
  defp normalize_data_list(data), do: [data]

  defp foldable_scalar_ast?(ast) when is_integer(ast) or is_float(ast), do: true
  defp foldable_scalar_ast?({:-, _, [v]}) when is_integer(v) or is_float(v), do: true
  defp foldable_scalar_ast?(_), do: false

  defp resolve_external_input(data_ast, state, scalar_fold?) do
    if scalar_fold? && foldable_scalar_ast?(data_ast) do
      {data_ast, state}
    else
      key = Macro.to_string(data_ast)

      case state.input_vars[key] do
        nil ->
          var_atom = String.to_atom("arg#{state.next_input_idx}")
          var_ast = {var_atom, [], nil}

          new_state = %{
            state
            | next_input_idx: state.next_input_idx + 1,
              input_vars: Map.put(state.input_vars, key, var_ast),
              input_order: state.input_order ++ [{data_ast, var_ast}]
          }

          {var_ast, new_state}

        var_ast ->
          {var_ast, state}
      end
    end
  end

  defp resolve_external_inputs([], state, _scalar_fold?), do: {[], state}

  defp resolve_external_inputs([data_ast | rest], state, scalar_fold?) do
    {arg_ast, state1} = resolve_external_input(data_ast, state, scalar_fold?)
    {other_asts, state2} = resolve_external_inputs(rest, state1, scalar_fold?)
    {[arg_ast | other_asts], state2}
  end

  defp collect_pattern_vars(ast, acc) do
    Macro.prewalk(ast, acc, fn
      {var, _meta, ctx} = node, vars
      when is_atom(var) and (is_atom(ctx) or is_nil(ctx)) ->
        {node, MapSet.put(vars, var)}

      node, vars ->
        {node, vars}
    end)
    |> elem(1)
  end

  defp collect_local_vars(body, param_vars) do
    Enum.reduce(List.wrap(body), MapSet.new(), fn node, acc ->
      case node do
        {:=, _, [lhs, _rhs]} ->
          MapSet.union(acc, collect_pattern_vars(lhs, MapSet.new()))

        {:type, _, [decl]} ->
          case decl do
            {var, _meta, _args} when is_atom(var) ->
              MapSet.put(acc, var)

            _ ->
              acc
          end

        _ ->
          acc
      end
    end)
    |> MapSet.difference(param_vars)
    |> MapSet.difference(MapSet.new([:return, :type]))
  end

  defp rename_local_vars(body, stage_idx, local_vars) do
    local_map =
      local_vars
      |> Enum.map(fn name -> {name, String.to_atom("s#{stage_idx}_#{name}")} end)
      |> Map.new()

    Macro.prewalk(body, fn
      {var, meta, ctx} when is_atom(var) and (is_atom(ctx) or is_nil(ctx)) ->
        case local_map[var] do
          nil -> {var, meta, ctx}
          renamed -> {renamed, meta, ctx}
        end

      node ->
        node
    end)
  end

  defp declared_var_from_type({var, _meta, _args}) when is_atom(var), do: var
  defp declared_var_from_type(_), do: nil

  defp substitute_expr({:=, meta, [lhs, rhs]}, param_map) do
    {:=, meta, [substitute_pattern(lhs, param_map), substitute_expr(rhs, param_map)]}
  end

  defp substitute_expr({:type, meta, [decl]}, param_map) do
    {:type, meta, [substitute_pattern(decl, param_map)]}
  end

  defp substitute_expr({var, _meta, ctx} = node, param_map)
       when is_atom(var) and (is_atom(ctx) or is_nil(ctx)) do
    case param_map[var] do
      nil -> node
      ast -> ast
    end
  end

  defp substitute_expr({form, meta, args}, param_map) when is_list(args) do
    {form, meta, Enum.map(args, &substitute_expr(&1, param_map))}
  end

  defp substitute_expr(list, param_map) when is_list(list) do
    Enum.map(list, &substitute_expr(&1, param_map))
  end

  defp substitute_expr(other, _param_map), do: other

  defp substitute_pattern({var, _meta, ctx} = node, _param_map)
       when is_atom(var) and (is_atom(ctx) or is_nil(ctx)) do
    case var do
      :_ -> node
      _ -> node
    end
  end

  defp substitute_pattern({form, meta, args}, param_map) when is_list(args) do
    {form, meta, Enum.map(args, &substitute_pattern(&1, param_map))}
  end

  defp substitute_pattern(list, param_map) when is_list(list) do
    Enum.map(list, &substitute_pattern(&1, param_map))
  end

  defp substitute_pattern(other, _param_map), do: other

  defp substitute_params(body, param_map) do
    {nodes, _final_map} =
      body
      |> List.wrap()
      |> Enum.map_reduce(param_map, fn node, env ->
        case node do
          {:=, meta, [lhs, rhs]} ->
            lhs_sub = substitute_pattern(lhs, env)
            rhs_sub = substitute_expr(rhs, env)

            next_env =
              lhs
              |> collect_pattern_vars(MapSet.new())
              |> Enum.reduce(env, fn var, acc -> Map.delete(acc, var) end)

            {{:=, meta, [lhs_sub, rhs_sub]}, next_env}

          {:type, meta, [decl]} ->
            decl_sub = substitute_pattern(decl, env)

            next_env =
              case declared_var_from_type(decl) do
                nil -> env
                var -> Map.delete(env, var)
              end

            {{:type, meta, [decl_sub]}, next_env}

          other ->
            {substitute_expr(other, env), env}
        end
      end)

    nodes
  end

  defp inline_kernel_with_args(kernel_ast, actual_args, stage_idx) do
    {formal_args, body} = decompose_kernel(kernel_ast)

    if length(formal_args) != length(actual_args) do
      raise ArgumentError,
            "fusion arity mismatch in stage #{stage_idx}: kernel expects #{length(formal_args)} args, got #{length(actual_args)}"
    end

    param_vars =
      formal_args
      |> Enum.map(fn {var, _meta, _ctx} -> var end)
      |> MapSet.new()

    local_vars = collect_local_vars(body, param_vars)
    renamed = rename_local_vars(body, stage_idx, local_vars)

    param_map =
      Enum.zip(formal_args, actual_args)
      |> Enum.map(fn {{name, _meta, _ctx}, arg_ast} -> {name, arg_ast} end)
      |> Map.new()

    substituted = substitute_params(renamed, param_map)
    split_body_and_return(substituted)
  end

  defp fuse_map_stage(call, stage_idx, state, scalar_fold?) do
    arity = stage_arity(call.ske)
    data_asts = normalize_data_list(call.data_ast)

    {actual_args, state1, stage_prelude} =
      if stage_idx == 0 do
        if length(data_asts) != arity do
          raise ArgumentError,
                "first stage #{call.ske} expects #{arity} explicit inputs, got #{length(data_asts)}"
        end

        {resolved, s} = resolve_external_inputs(data_asts, state, scalar_fold?)
        {resolved, s, []}
      else
        if length(data_asts) != arity - 1 do
          raise ArgumentError,
                "stage #{stage_idx + 1} #{call.ske} expects #{arity - 1} explicit inputs in a pipe chain, got #{length(data_asts)}"
        end

        {extra_args, state_after_inputs} = resolve_external_inputs(data_asts, state, scalar_fold?)
        input_var = {String.to_atom("__fuse_in_#{stage_idx}"), [], nil}
        prelude = [{:=, [], [input_var, state.value_ast]}]
        {[input_var | extra_args], state_after_inputs, prelude}
      end

    {stage_prefix, stage_value} = inline_kernel_with_args(call.kernel_ast, actual_args, stage_idx)

    %{
      state1
      | body: state1.body ++ stage_prelude ++ stage_prefix,
        value_ast: stage_value
    }
  end

  defp emit_map_from_inputs_and_fun([t1], fun_ast) do
    quote do
      Ske.map(unquote(t1), unquote(fun_ast))
    end
  end

  defp emit_map_from_inputs_and_fun([t1, t2], fun_ast) do
    quote do
      Ske.map2(unquote(t1), unquote(t2), unquote(fun_ast))
    end
  end

  defp emit_map_from_inputs_and_fun([t1, t2, t3], fun_ast) do
    quote do
      Ske.map3(unquote(t1), unquote(t2), unquote(t3), unquote(fun_ast))
    end
  end

  defp emit_map_from_inputs_and_fun(inputs, _fun_ast) do
    raise ArgumentError,
          "full-chain fusion requires <= 3 tensor inputs with current Ske API, got #{length(inputs)}"
  end

  defp fuse_map_chain_calls(calls, scalar_fold?) do
    if Enum.empty?(calls) do
      raise ArgumentError, "empty fusion chain"
    end

    invalid = Enum.find(calls, fn call -> call.ske not in [:map, :map2, :map3] end)

    if invalid do
      raise ArgumentError,
            "full-chain map fusion supports only map/map2/map3 calls, got #{inspect(invalid.ske)}"
    end

    state0 = %{body: [], value_ast: nil, next_input_idx: 0, input_vars: %{}, input_order: []}

    state =
      calls
      |> Enum.with_index()
      |> Enum.reduce(state0, fn {call, idx}, acc ->
        fuse_map_stage(call, idx, acc, scalar_fold?)
      end)

    input_data_asts = Enum.map(state.input_order, fn {data_ast, _var_ast} -> data_ast end)
    input_vars = Enum.map(state.input_order, fn {_data_ast, var_ast} -> var_ast end)
    fused_fun = build_phok_fun(input_vars, state.body ++ [state.value_ast])

    %{inputs: input_data_asts, fun: fused_fun}
  end

  defp emit_single_call(%AstCall{ske: :map, data_ast: nil}) do
    raise ArgumentError,
          "standalone map fusion expects Ske.map(input, kernel)"
  end

  defp emit_single_call(%AstCall{ske: :map, data_ast: data_ast, kernel_ast: kernel_ast}) do
    quote do
      Ske.map(unquote(data_ast), unquote(normalize_kernel_ast(kernel_ast)))
    end
  end

  defp emit_single_call(%AstCall{ske: :map2, data_ast: [t1, t2], kernel_ast: kernel_ast}) do
    quote do
      Ske.map2(unquote(t1), unquote(t2), unquote(normalize_kernel_ast(kernel_ast)))
    end
  end

  defp emit_single_call(%AstCall{ske: :map3, data_ast: [t1, t2, t3], kernel_ast: kernel_ast}) do
    quote do
      Ske.map3(unquote(t1), unquote(t2), unquote(t3), unquote(normalize_kernel_ast(kernel_ast)))
    end
  end

  defp emit_single_call(%AstCall{ske: :reduce, data_ast: data_ast, kernel_ast: kernel_ast}) do
    case data_ast do
      [input, initial] ->
        quote do
          Ske.reduce(unquote(input), unquote(initial), unquote(normalize_kernel_ast(kernel_ast)))
        end

      _ ->
        raise ArgumentError,
              "standalone reduce fusion expects Ske.reduce(input, initial, kernel)"
    end
  end

  defp emit_single_call(%AstCall{} = call) do
    raise ArgumentError,
          "unsupported single fusion call shape for #{inspect(call.ske)} with data #{Macro.to_string(call.data_ast)}"
  end

  defp emit_chain_with_reduce(calls, scalar_fold?) do
    reduce_idx =
      calls
      |> Enum.with_index()
      |> Enum.find_value(fn
        {%AstCall{ske: :reduce}, idx} -> idx
        _ -> nil
      end)

    map_calls = Enum.take(calls, reduce_idx)
    reduce_call = Enum.at(calls, reduce_idx)

    if Enum.empty?(map_calls) do
      emit_single_call(reduce_call)
    else
      %{inputs: inputs, fun: map_fun} = fuse_map_chain_calls(map_calls, scalar_fold?)
      red_fun = normalize_kernel_ast(reduce_call.kernel_ast)
      initial = reduce_call.data_ast

      case inputs do
        [t1] ->
          quote do
            Ske.mapReduce(unquote(t1), unquote(initial), unquote(map_fun), unquote(red_fun))
          end

        [t1, t2] ->
          quote do
            Ske.map2Reduce(
              unquote(t1),
              unquote(t2),
              unquote(initial),
              unquote(map_fun),
              unquote(red_fun)
            )
          end

        _ ->
          raise ArgumentError,
                "map-chain |> reduce fusion currently supports fused map arity up to 2, got #{length(inputs)}"
      end
    end
  end

  defp emit_fused_chain(calls, scalar_fold?) do
    reduce_positions =
      calls
      |> Enum.with_index()
      |> Enum.filter(fn {%AstCall{ske: ske}, _idx} -> ske == :reduce end)
      |> Enum.map(fn {_call, idx} -> idx end)

    case reduce_positions do
      [] ->
        %{inputs: inputs, fun: fun_ast} = fuse_map_chain_calls(calls, scalar_fold?)
        emit_map_from_inputs_and_fun(inputs, fun_ast)

      [idx] when idx == length(calls) - 1 ->
        emit_chain_with_reduce(calls, scalar_fold?)

      [idx] ->
        raise ArgumentError,
              "reduce must be the final stage in fusion chains, found at stage #{idx + 1}"

      _ ->
        raise ArgumentError, "fusion chain supports at most one reduce stage"
    end
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
         {:&, _, [{:/, _, [{{:., _, [{:__aliases__, _, _}, f_name]}, _, []}, _f_arity]}]}
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

  defmacro {{:.,_,[_, :spawn]},_,call_lhs} <~> {{:.,module_line,[aliases_tuple, :spawn]}, line, call_rhs} do

    [kernel_call, _, _, _] = call_lhs
    IO.inspect(kernel_call)
    kernel_name = JIT.get_kernel_name(kernel_call)
    kast = PolyHok.load_ast(kernel_name)
    IO.inspect(kast, label: "inspecting kernel")
   {{:.,module_line,[aliases_tuple, :spawn]}, line, call_rhs}
  end

  defmacro with_fusion(ast, opts \\ []) do
    opts =
      case opts do
        list when is_list(list) ->
          Keyword.validate!(list, scalar_fold: true)

        other ->
          raise ArgumentError,
                "Fusion.with_fusion/2 expects a keyword options list, got: #{Macro.to_string(other)}"
      end

    scalar_fold? = Keyword.fetch!(opts, :scalar_fold)

    calls =
      ast
      |> flatten_pipe_ast()
      |> Enum.map(&parse_ske_call/1)

    case calls do
      [single] -> emit_single_call(single)
      chain -> emit_fused_chain(chain, scalar_fold?)
    end
  end
end
