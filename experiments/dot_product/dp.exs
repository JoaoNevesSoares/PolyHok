require PolyHok
require Fusion
use Ske

PolyHok.defmodule Dp do
  defd add(x, y) do
    x + y
  end

  def dot_unfs(x, y, initial) do
    res =
      Ske.map2(x, y, PolyHok.phok(fn x, y -> x * y end))
      |> Ske.reduce(initial, &Dp.add/2)

    res
  end

  def dot_fs(x, y, initial) do
    res =
      Fusion.with_fusion(
        Ske.map2(x, y, PolyHok.phok(fn x, y -> x * y end))
        |> Ske.reduce(initial, &Dp.add/2)
      )
    res
  end
end
