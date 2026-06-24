require PolyHok
require Fusion
use Ske

PolyHok.defmodule Axpy do
  defd axpy(x, y) do
    3.1415 * x + y
  end

  def unfused(a, b, c) do
    Ske.map2(a, b, &Axpy.axpy/2)
    |> Ske.map2(c, &Axpy.axpy/2)
  end

  def fused(a, b, c) do
      Fusion.with_fusion(
        Ske.map2(a, b, &Axpy.axpy/2)
        |> Ske.map2(c, &Axpy.axpy/2)
      )
  end
end
