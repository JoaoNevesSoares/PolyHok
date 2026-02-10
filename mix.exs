defmodule PolyHok.MixProject do
  use Mix.Project

  def project do
    [
      app: :poly_hok,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  defp deps do
    [
      {:matrex, "~> 0.6"},
      {:nx, "~> 0.9.2"},
      {:mix_unused, "~> 0.4.0"}
    ]
  end
end
