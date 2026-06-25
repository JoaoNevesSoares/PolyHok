Code.require_file("ni.exs", __DIR__)

defmodule Nibench do
  def run(:fused, ver) do
    IO.inspect(ver)
    start_time = System.monotonic_time()
    res =
      case ver do
        "v2" ->
          Ni.ni_fs_v2()

        "v3" ->
          Ni.ni_fs_v3()

        _ ->
          Ni.ni_fs()
      end
    end_time = System.monotonic_time()
    IO.inspect(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def run(:unfused, ver) do
    IO.inspect(ver)
    start_time = System.monotonic_time()
    res =
      case ver do
        "v2" ->
          Ni.ni_unfs_v2()

        "v3" ->
          Ni.ni_unfs_v3()
        _ ->
          Ni.ni_unfs()
      end
    end_time = System.monotonic_time()
    IO.inspect(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def main() do
    argv = System.argv()

    {[{:fun, ver} | _rest], [], []} =
      OptionParser.parse(argv, strict: [fun: :string])

    file = File.open!("ni_#{ver}.csv", [:write, :utf8])

    res =
      Enum.reduce(1..30, [], fn _, acc ->
        fs = run(:fused, ver)
        ufs = run(:unfused, ver)

        acc ++ [%{"unfused" => ufs, "fused" => fs}]
      end)

    res
    |> CSV.encode(headers: ["unfused", "fused"])
    |> Enum.each(&IO.write(file, &1))

    File.close(file)
  end
end

Nibench.main()
