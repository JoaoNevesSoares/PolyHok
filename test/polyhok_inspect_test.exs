defmodule PolyHokInspectTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  test "block_inspect prints fused anonymous functions as PolyHok.phok calls" do
    module = Module.concat(__MODULE__, "Sample#{System.unique_integer([:positive])}")

    ast =
      quote do
        defmodule unquote(module) do
          require Fusion
          require PolyHokInspect

          def run(dev_x) do
            PolyHokInspect.block_inspect do
              Fusion.with_fusion(
                Ske.map(dev_x, PolyHok.phok(fn x -> x * 3.0 end))
                |> Ske.map(PolyHok.phok(fn y -> y + 1.0 end))
              )
            end
          end
        end
      end

    output = capture_io(fn -> Code.compile_quoted(ast) end)
    normal_output = output |> String.split("Codigo normal:") |> List.last()

    assert normal_output =~ "PolyHok.phok(fn"
    assert normal_output =~ "Ske.map("
    refute normal_output =~ "{:anon,"
    refute normal_output =~ "{{:fn,"
  end
end
