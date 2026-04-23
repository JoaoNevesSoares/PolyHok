require PolyHok

PolyHok.defmodule NBodies do

  defd body_interaction(bi, bj, ai) do
    float3 r;

    r.x = bj[0] - bi[0];
    r.y = bj[1] - bi[1];
    r.z = bj[2] - bi[2];

    float dist_sqr = r.x * r.x + r.y * r.y + r.z * r.z + 0.000000001; 
    float inv_dist = rsqrtf(dist_sqr);
    float inv_dist3 = inv_dist * inv_dist * inv_dist;

    float s = bj[3] * inv_dist3;
    ai[0] = ai[0] + r.x * s;
    ai[1] = ai[1] + r.y * s;
    ai[2] = ai[2] + r.z * s;
  end
end
use Ske

# n = 10

h_arr = Nx.tensor([6.0, 7.0, 8.0], type: {:f, 32})

b_arr = Nx.tensor(Enum.to_list(1..3), type: {:f, 32})
c_arr = Nx.tensor(Enum.to_list(1..3), type: {:f, 32})
IO.inspect(c_arr)
IO.inspect(h_arr)
gpu_arr = PolyHok.new_gnx(h_arr)
b_dev = PolyHok.new_gnx(b_arr)
c_dev = PolyHok.new_gnx(c_arr)

gpu_resp = Ske.map(gpu_arr,&NBodies.body_interaction/3,[b_dev, c_dev], return: false)

resp = PolyHok.get_gnx(gpu_resp)
IO.inspect(resp)
