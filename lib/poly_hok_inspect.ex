defmodule PolyHokInspect do
  defp expand_all_macro_ast(ast, %Macro.Env{} = env) do
    Macro.prewalk(ast, fn node ->
      Macro.expand(node, env)
    end)
  end

  defp normalize_display_ast(ast) do
    Macro.prewalk(ast, &normalize_display_node/1)
  end

  defp normalize_display_node({:{}, _, [:anon, name, {escaped_function_ast, _funs}]} = node)
       when is_binary(name) do
    function_ast = unescape_tuple_ast(escaped_function_ast)

    if fn_ast?(function_ast) do
      phok_display_call(function_ast)
    else
      node
    end
  end

  defp normalize_display_node({:anon, name, {function_ast, _funs}} = node) when is_binary(name) do
    if fn_ast?(function_ast) do
      phok_display_call(function_ast)
    else
      node
    end
  end

  defp normalize_display_node(node), do: node

  defp unescape_tuple_ast({:{}, _, values}) do
    values
    |> Enum.map(&unescape_tuple_ast/1)
    |> List.to_tuple()
  end

  defp unescape_tuple_ast(values) when is_list(values),
    do: Enum.map(values, &unescape_tuple_ast/1)

  defp unescape_tuple_ast(value), do: value

  defp fn_ast?({:fn, _, [{:->, _, [args, _body]}]}) when is_list(args), do: true
  defp fn_ast?(_), do: false

  defp phok_display_call(function_ast) do
    quote do
      PolyHok.phok(unquote(function_ast))
    end
  end

  defmacro block_inspect(do: block) do
    original_ast = block
    expanded_once = Macro.expand_once(block, __CALLER__)
    expanded_ast = expand_all_macro_ast(block, __CALLER__)
    original_ast_pretty = inspect(original_ast, pretty: true, limit: :infinity)
    expanded_ast_pretty = inspect(expanded_ast, pretty: true, limit: :infinity)

    generated_code =
      expanded_ast
      |> normalize_display_ast()
      |> Macro.to_string()
      |> Code.format_string!()
      |> IO.iodata_to_binary()

    generated_code_normal =
      expanded_once
      |> normalize_display_ast()
      |> Macro.to_string()

    IO.puts("""

    \n=== MacroDbg ===
    -- AST (original):
    #{original_ast_pretty}

    -- AST (expandida):
    #{expanded_ast_pretty}

    -- Código gerado (após expansão):
    #{generated_code}
    === /MacroDbg ===

    """)

    IO.puts("Codigo normal:
      #{generated_code_normal}
      ")

    block
  end

  defmacro block_inspect_when_polyhok(do: block) do
    original_ast = block
    expanded_once = Macro.expand_once(block, __CALLER__)
    expanded_ast = expand_all_macro_ast(block, __CALLER__)
    original_ast_pretty = inspect(original_ast, pretty: true, limit: :infinity)
    expanded_ast_pretty = inspect(expanded_ast, pretty: true, limit: :infinity)

    generated_code =
      expanded_ast
      |> normalize_display_ast()
      |> Macro.to_string()

    generated_code_normal =
      expanded_once
      |> normalize_display_ast()
      |> Macro.to_string()

    IO.puts("""

    \n=== MacroDbg ===
    -- AST (original):
    #{original_ast_pretty}

    -- AST (expandida):
    #{expanded_ast_pretty}

    -- Código gerado (após expansão):
    #{generated_code}
    === /MacroDbg ===

    """)

    IO.puts("Codigo normal:
      #{generated_code_normal}
      ")

    block
  end

  defmacro left <~> _right do
    expanded = Macro.expand(left, __CALLER__)
    IO.inspect(expanded, label: "expanding my macro")
    expanded
  end
end
