defmodule PolyHok.MixProject do
  use Mix.Project

  def project do
    [
      app: :polyhok,
      version: "1.0.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      consolidate_protocols: true,
      deps: deps()
    ]
  end

  defp deps do
    [
      {:matrex, "~> 0.6"},
      {:nx, "~> 0.9.2"},
      {:csv, "~> 3.2"}
    ]
  end
end
