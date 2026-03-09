defmodule PMap do
  @threads_per_block 128

  # Uses the precompiled Elixir.PMap.so kernel (map2_call).
  # For unary map semantics we pass the same input tensor twice.
  def map({:nx, {:f, 32}, shape, _names, in_ref}, {:anon, _name, {lambda_ast, _deps}}) do
    out = PolyHok.new_gnx(shape, {:f, 32})
    {:nx, _out_type, _out_shape, _out_names, out_ref} = out

    size = Tuple.product(shape)
    blocks = div(size + @threads_per_block - 1, @threads_per_block)

    fun_ref = compile_lambda_ptr(lambda_ast)
    kernel_ref = PolyHok.load_kernel_nif(~c"Elixir.PMap", ~c"map2")

    PolyHok.spawn_nif(kernel_ref, {blocks, 1, 1}, {@threads_per_block, 1, 1}, [
      in_ref,
      in_ref,
      out_ref,
      size,
      fun_ref
    ])

    out
  end

  def map({:nx, type, _shape, _names, _ref}, _func) do
    raise "PMap.map/2 currently only supports {:f, 32} tensors, got #{inspect(type)}"
  end

  def map(other, _func) do
    raise "PMap.map/2 expects a GNX tensor, got #{inspect(other)}"
  end

  def comp_func(_arr1, _arr2, _n, _func) do
    raise "PMap.comp_func/4 is not implemented in this runtime module"
  end

  defp compile_lambda_ptr(lambda_ast) do
    {code, lambda_name} = compile_lambda_code(lambda_ast)

    module_name = "Elixir.PMapRuntime_#{System.unique_integer([:positive])}"
    cu_path = "c_src/#{module_name}.cu"
    so_path = "priv/#{module_name}.so"

    File.write!(cu_path, code)

    {result, errcode} =
      System.cmd(
        "nvcc",
        [
          "--shared",
          "--compiler-options",
          "'-fPIC'",
          "-o",
          so_path,
          cu_path
        ],
        stderr_to_stdout: true
      )

    if errcode in [1, 2] do
      raise "Error compiling runtime lambda for PMap:\n#{result}"
    end

    PolyHok.load_fun_nif(String.to_charlist(module_name), String.to_charlist(lambda_name))
  end

  defp compile_lambda_code(lambda_ast) do
    suffix = PolyHok.CudaBackend.gen_lambda_name()
    lambda_name = "anonymous_#{suffix}"
    lambda_types = [:float, :float, :float]

    {code, _fun_type} = PolyHok.CudaBackend.compile_lambda(lambda_ast, lambda_types, suffix, "PMap")
    {code, lambda_name}
  end
end
