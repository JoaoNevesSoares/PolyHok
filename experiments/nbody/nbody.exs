require PolyHok
import BoundAnalysis

PolyHok.defmodule NBody do
  def generate_bodies(num_bodies, type) do
    PolyHok.random_gnx(1, 2, num_bodies * 4, type)
  end

  def generate_accelerations(num_bodies, type) do
    PolyHok.random_gnx(1, 2,num_bodies * 3, type)
  end

  def generate_velocities(num_bodies, type) do
    PolyHok.random_gnx(1, 2,num_bodies * 3, type)
  end

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

  def run_spawn(num_bodies, positions, accel, velocities) do
    threadsPerBlock = 128
    blocksPerGrid = div(num_bodies + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&NBody.calculate_forces/3,
      {blocksPerGrid, 1, 1},
      {threadsPerBlock, 1, 1},
      [positions, accel, num_bodies]
    )

    PolyHok.spawn(&NBody.calculate_velocities/4, 
      {blocksPerGrid, 1, 1}, 
      {threadsPerBlock, 1, 1}, 
      [velocities, accel, 0.01, num_bodies] 
    )

    PolyHok.spawn(&NBody.calculate_positions/4, 
      {blocksPerGrid, 1, 1}, 
      {threadsPerBlock, 1, 1}, 
      [positions, velocities, 0.01, num_bodies]
    )
    positions
  end

  def run_fused_spawn(num_bodies, positions, accel, velocities) do
    threadsPerBlock = 128
    blocksPerGrid = div(num_bodies + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&NBody.calculate_forces/3,
      {blocksPerGrid, 1, 1},
      {threadsPerBlock, 1, 1},
      [positions, accel, num_bodies]
    )
    fuse(
      PolyHok.spawn(&NBodies.calculate_velocities/4,
      {blocksPerGrid, 1, 1}, 
      {threadsPerBlock, 1, 1},
      [velocities, accel, 0.01, num_bodies]
      ),
      PolyHok.spawn(&NBodies.calculate_positions/4,
      {blocksPerGrid, 1, 1}, 
      {threadsPerBlock, 1, 1},
      [positions, velocities, 0.01, num_bodies])
    )
    positions
  end
end
