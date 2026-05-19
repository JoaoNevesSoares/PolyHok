require PolyHok

PolyHok.defmodule NBodies do
  defk calculate_forces(bodies, accel, num_bodies) do
    tid = blockIdx.x * blockDim.x + threadIdx.x

    if tid < num_bodies do
      ax = 0.0
      ay = 0.0
      az = 0.0

      for j in range(0, num_bodies) do
        rx = bodies[j * 4 + 0] - bodies[tid * 4 + 0]
        ry = bodies[j * 4 + 1] - bodies[tid * 4 + 1]
        rz = bodies[j * 4 + 2] - bodies[tid * 4 + 2]
        dist_sqr = rx * rx + ry * ry + rz * rz + 0.000000001
        inv_dist = rsqrtf(dist_sqr)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        s = bodies[j * 4 + 3] * inv_dist3
        ax = ax + rx * s
        ay = ay + ry * s
        az = az + rz * s
      end

      accel[tid * 3 + 0] = ax
      accel[tid * 3 + 1] = ay
      accel[tid * 3 + 2] = az
    end
  end

  defk calculate_velocities(velocities, accel, dt, num_bodies) do 
    tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid <= num_bodies do
      velocities[tid * 3 + 0] = velocities[tid * 3 + 0] + accel[tid * 3 + 0] * dt
      velocities[tid * 3 + 1] = velocities[tid * 3 + 1] + accel[tid * 3 + 1] * dt
      velocities[tid * 3 + 2] = velocities[tid * 3 + 2] + accel[tid * 3 + 2] * dt
    end
  end
  defk calculate_positions(positions, velocities, dt, num_bodies) do 
    tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid <= num_bodies do 
      positions[tid * 4 + 0] = positions[tid * 4 + 0] + velocities[tid * 3 + 0] * dt
      positions[tid * 4 + 1] = positions[tid * 4 + 1] + velocities[tid * 3 + 1] * dt
      positions[tid * 4 + 2] = positions[tid * 4 + 2] + velocities[tid * 3 + 2] * dt
    end
  end
end

pos_body =
  Nx.tensor(
    [
      600.0,
      700.0,
      800.0,
      1.0,
      601.0,
      700.0,
      800.0,
      1.0,
      600.0,
      701.0,
      800.0,
      1.0
    ],
    type: {:f, 32}
  )

accel_body =
  Nx.tensor(List.duplicate(0.0, 9), type: {:f, 32})

vel_body =
  Nx.tensor(List.duplicate(0.0, 9), type: {:f, 32})

positions = PolyHok.new_gnx(pos_body)

accel = PolyHok.new_gnx(accel_body)
velocities = PolyHok.new_gnx(vel_body)

PolyHok.spawn(
  &NBodies.calculate_forces/3,
  {1, 1, 1},
  {3, 1, 1},
  [positions, accel, 3]
)
PolyHok.spawn(&NBodies.calculate_velocities/4, {1,1,1}, {3,1,1}, [velocities, accel, 0.01, 3])
PolyHok.spawn(&NBodies.calculate_positions/4, {1,1,1}, {3,1,1}, [positions, velocities, 0.01, 3])

res_pos = PolyHok.get_gnx(positions)
res_vel = PolyHok.get_gnx(velocities)

IO.inspect(res_pos)
IO.inspect(res_vel)
