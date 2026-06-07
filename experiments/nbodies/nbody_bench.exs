require PolyHok

defmodule NBodyInput do
  def generate_bodies(num_bodies) do
    random_tensor(num_bodies * 4)
  end

  def generate_accelerations(num_bodies) do
    random_tensor(num_bodies * 3)
  end

  def generate_velocities(num_bodies) do
    random_tensor(num_bodies * 3)
  end

  defp random_tensor(count) do
    values = for _ <- 1..count, do: :rand.uniform()

    Nx.tensor(values, type: {:f, 32})
  end
end

PolyHok.defmodule NBodies do
  import BoundAnalysis
  defk calculate_forces(bodies, accel, num_bodies) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    a[3]

    if tid < num_bodies do
      a[0] = 0.0
      a[1] = 0.0
      a[2] = 0.0

      for j in range(0, num_bodies) do
        rx = bodies[j * 4 + 0] - bodies[tid * 4 + 0]
        ry = bodies[j * 4 + 1] - bodies[tid * 4 + 1]
        rz = bodies[j * 4 + 2] - bodies[tid * 4 + 2]
        dist_sqr = rx * rx + ry * ry + rz * rz + 0.000000001
        inv_dist = rsqrtf(dist_sqr)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        s = bodies[j * 4 + 3] * inv_dist3
        a[0] = a[0] + rx * s
        a[1] = a[1] + ry * s
        a[2] = a[2] + rz * s
      end

      accel[tid * 3 + 0] = a[0]
      accel[tid * 3 + 1] = a[1]
      accel[tid * 3 + 2] = a[2]
    end
  end

  defk calculate_velocities(velocities, accel, dt, num_bodies) do
    tid = blockIdx.x * blockDim.x + threadIdx.x
    velocities[tid * 3 + 0] = velocities[tid * 3 + 0] + accel[tid * 3 + 0] * dt
    velocities[tid * 3 + 1] = velocities[tid * 3 + 1] + accel[tid * 3 + 1] * dt
    velocities[tid * 3 + 2] = velocities[tid * 3 + 2] + accel[tid * 3 + 2] * dt
  end

  defk calculate_positions(positions, velocities, dt, num_bodies) do
    tid = blockIdx.x * blockDim.x + threadIdx.x
    positions[tid * 4 + 0] = positions[tid * 4 + 0] + velocities[tid * 3 + 0] * dt
    positions[tid * 4 + 1] = positions[tid * 4 + 1] + velocities[tid * 3 + 1] * dt
    positions[tid * 4 + 2] = positions[tid * 4 + 2] + velocities[tid * 3 + 2] * dt
  end

  def run_unfused() do
    num_bodies = 262_144
    pos_body = NBodyInput.generate_bodies(num_bodies)
    accel_body = NBodyInput.generate_accelerations(num_bodies)
    vel_body = NBodyInput.generate_velocities(num_bodies)
    positions = PolyHok.new_gnx(pos_body)
    accel = PolyHok.new_gnx(accel_body)
    velocities = PolyHok.new_gnx(vel_body)
    prev = System.monotonic_time()

    PolyHok.spawn(
      &NBodies.calculate_forces/3,
      {2048, 1, 1},
      {128, 1, 1},
      [positions, accel, num_bodies]
    )

    PolyHok.spawn(&NBodies.calculate_velocities/4, {2048, 1, 1}, {128, 1, 1}, [
      velocities,
      accel,
      0.01,
      num_bodies
    ])

    PolyHok.spawn(&NBodies.calculate_positions/4, {2048, 1, 1}, {128, 1, 1}, [
      positions,
      velocities,
      0.01,
      num_bodies
    ])

    res_pos = PolyHok.get_gnx(positions)
    next = System.monotonic_time()
    IO.inspect(res_pos)
    IO.puts("PolyHok\t#{num_bodies}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
  end

  def run_fused() do
    num_bodies = 524288
    pos_body = NBodyInput.generate_bodies(num_bodies)
    accel_body = NBodyInput.generate_accelerations(num_bodies)
    vel_body = NBodyInput.generate_velocities(num_bodies)
    positions = PolyHok.new_gnx(pos_body)
    accel = PolyHok.new_gnx(accel_body)
    velocities = PolyHok.new_gnx(vel_body)
    prev = System.monotonic_time()

    PolyHok.spawn(
      &NBodies.calculate_forces/3,
      {4096, 1, 1},
      {128, 1, 1},
      [positions, accel, num_bodies]
    )

    BoundAnalysis.fuse(
      PolyHok.spawn(&NBodies.calculate_velocities/4, {4096, 1, 1}, {128, 1, 1}, [
        velocities,
        accel,
        0.01,
        num_bodies
      ]),
      PolyHok.spawn(&NBodies.calculate_positions/4, {4096, 1, 1}, {128, 1, 1}, [
        positions,
        velocities,
        0.01,
        num_bodies
      ])
    )

    res_pos = PolyHok.get_gnx(positions)
    next = System.monotonic_time()
    # IO.inspect(res_pos)
    IO.puts("PolyHok\t#{num_bodies}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
  end
end

Enum.each(1..10, fn x -> 
  # NBodies.run_unfused()
  NBodies.run_fused()
end)
