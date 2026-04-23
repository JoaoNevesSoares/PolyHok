defmodule PolyHokInspect do
  defp expand_all_macro_ast(ast, %Macro.Env{} = env) do
    Macro.prewalk(ast, fn node ->
      Macro.expand(node, env)
    end)
  end

  defmacro block_inspect(do: block) do
    original_ast = block
    expanded_once = Macro.expand_once(block, __CALLER__)
    expanded_ast = expand_all_macro_ast(block, __CALLER__)
    original_ast_pretty = inspect(original_ast, pretty: true, limit: :infinity)
    expanded_ast_pretty = inspect(expanded_ast, pretty: true, limit: :infinity)

    generated_code =
      expanded_ast
      |> Macro.to_string()
      |> Code.format_string!()
      |> IO.iodata_to_binary()

    generated_code_normal = Macro.to_string(expanded_once)

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
      |> Macro.to_string()

    generated_code_normal = Macro.to_string(expanded_once)

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
