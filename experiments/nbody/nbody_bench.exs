Code.require_file("nbody.exs", __DIR__)

defmodule NBodybench do
  def run(:fused, num_bodies, positions, accel, velocities) do
    start_time = System.monotonic_time()
    res = NBody.run_fused_spawn(num_bodies, positions, accel, velocities)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def run(:unfused, num_bodies, positions, accel, velocities) do
    start_time = System.monotonic_time()
    res = NBody.run_spawn(num_bodies, positions, accel, velocities)
    end_time = System.monotonic_time()
    PolyHok.get_gnx(res)
    System.convert_time_unit(end_time - start_time, :native, :millisecond)
  end

  def gen_data(n, type) do
    pos = NBody.generate_bodies(n, type)
    accel = NBody.generate_accelerations(n,type)
    vel = NBody.generate_velocities(n, type)
    {pos, accel, vel}
  end

  def main() do
    argv = System.argv()

    {[{:size, n}, {:type, type}, {:order, ord} | _rest], [], []} =
      OptionParser.parse(argv, strict: [size: :integer, type: :string, order: :integer])

    file = File.open!("results/nbody/nbodies_#{type}_#{ord}_#{n}.csv", [:write, :utf8])

     {pos, accel, vel} = case type do
        "float" -> gen_data(n, {:f, 32})
        "double" -> gen_data(n, {:f, 64})
        "integer" -> gen_data(n, {:s, 32})
      end

    start = ord
    finish = 29 + ord
    res =
      Enum.reduce(start..finish, [], fn i, acc ->
        if rem(i, 2) == 0 do
          fs = run(:fused, n, pos, accel, vel)
          ufs = run(:unfused, n, pos, accel, vel)
          acc ++ [%{"unfused" => ufs, "fused" => fs}]
        else
          ufs = run(:unfused, n, pos, accel, vel)
          fs = run(:fused, n, pos, accel, vel)
          acc ++ [%{"unfused" => ufs, "fused" => fs}]
        end
      end)

    res
    |> CSV.encode(headers: ["unfused", "fused"])
    |> Enum.each(&IO.write(file, &1))

    File.close(file)
  end
end

NBodybench.main()
