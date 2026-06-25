require PolyHok
require Fusion
use Ske

PolyHok.defmodule Dp do
  defd add(x, y) do
    x + y
  end

  def dot_unfs(x, y) do
    res =
      Ske.map2(x, y, PolyHok.phok(fn x, y -> x * y end))
      |> Ske.reduce(0.0, &Dp.sum/2)

    res
  end

  def dot_fs(x, y) do
    res =
      Fusion.with_fusion(
        Ske.map2(x, y, PolyHok.phok(fn x, y -> x * y end))
        |> Ske.reduce(0.0, &Dp.sum/2)
      )
    res
  end
end
