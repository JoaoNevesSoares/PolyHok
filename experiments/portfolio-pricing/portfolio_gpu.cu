#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err),    \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const int OPT_N = 1 << 20;
static const size_t OPT_SZ = (size_t)OPT_N * sizeof(float);

static const float RISKFREE = 0.02f;
static const float VOLATILITY = 0.30f;

float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

__device__ inline float norm_cdf(float x) {

  const float a1 = 0.31938153f;
  const float a2 = -0.356563782f;
  const float a3 = 1.781477937f;
  const float a4 = -1.821255978f;
  const float a5 = 1.330274429f;
  const float rsqrt_2pi = 0.39894228040143267793994605993438f;
  const float p = 0.2316419f;

  float ax = fabsf(x);
  float k = 1.0f / (1.0f + p * ax);
  float poly = (((((a5 * k + a4) * k + a3)) * k + a2) * k + a1) * k;
  float phi = rsqrt_2pi * expf(-0.5f * ax * ax);
  float cnd = 1.0f - phi * poly;

  if (x < 0)
    cnd = 1.0f - cnd;
  return cnd;
}

__device__ float warp_reduce_sum(float v) {

  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

/*
 * S -> Stock price
 * X -> Option strike
 * T -> Option years
 * R -> Riskless rate
 * V -> Volatility rate
 */
__device__ float blackscholes_body(float S, float X, float T, float r,
                                   float v) {

  float sqrtT = sqrtf(T);
  float d1 = (logf(S / X) + (r + (v * v) / 2.0f) * T) / (v * sqrtT);
  float d2 = d1 - v * sqrtT;
  return S * norm_cdf(d1) - X * __expf(-r * T) * norm_cdf(d2);
}

__global__ void black_scholes(float *call_result, float *stock_price,
                              float *option_strike, float *option_years,
                              float riskfree, float volatility, int optN) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int opt = tid; opt < optN; opt += stride) {
    call_result[opt] =
        blackscholes_body(stock_price[opt], option_strike[opt],
                          option_years[opt], riskfree, volatility);
  }
}

__global__ void weightMultiply(float *price, float *weight, float *contrib,
                               int optN) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < optN; i += stride) {
    contrib[i] = price[i] * weight[i];
  }
}

__global__ void reduce_sum_blocks(float *out, float *in, int N) {

  extern __shared__ float smem[];

  int tid_global = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  float sum = 0.0f;
  for (int i = tid_global; i < N; i += stride) {
    sum += in[i];
  }

  smem[threadIdx.x] = sum;
  __syncthreads();

  // shared reduce down to 32
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s)
      smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }

  // final warp reduce

  if (threadIdx.x < 32) {
    float v = smem[threadIdx.x];
    if (blockDim.x >= 64)
      v += smem[threadIdx.x + 32];
    v = warp_reduce_sum(v);
    if (threadIdx.x == 0)
      out[blockIdx.x] = v;
  }
}

// GPU reduce to 1 element using ping-pong buffers.
// Returns a device pointer to a single float (size 1) located in either bufA or
// bufB.
static float *reduce_to_one_gpu(float *d_in, int N, int threads,
                                int blocksReduce, float *bufA, float *bufB) {
  const size_t shmem = (size_t)threads * sizeof(float);

  // First pass: input is d_in, output is bufA (size blocksReduce)
  reduce_sum_blocks<<<blocksReduce, threads, shmem>>>(bufA, d_in, N);
  CUDA_CHECK(cudaGetLastError());

  int curN = blocksReduce;
  float *curIn = bufA;
  float *curOut = bufB;

  // Subsequent passes: reduce curN down until 1
  while (curN > 1) {
    int nextBlocks = (curN + threads - 1) / threads;
    if (nextBlocks > blocksReduce)
      nextBlocks = blocksReduce;
    if (nextBlocks < 1)
      nextBlocks = 1;

    reduce_sum_blocks<<<nextBlocks, threads, shmem>>>(curOut, curIn, curN);
    CUDA_CHECK(cudaGetLastError());

    curN = nextBlocks;

    // ping-pong swap
    const float *tmpIn = curIn;
    curIn = curOut;
    curOut = (float *)tmpIn;
  }

  // curIn points to buffer that contains 1 element
  return (float *)curIn;
}

int main(int argc, char **argv) {

  cudaEvent_t start_timer;
  cudaEvent_t stop_timer;
  
  float elapsed_time;
  // Host buffers
  float *call_price, *stock_price, *option_strike, *option_years, *pt_weight;

  // Device buffers
  float *d_call_price, *d_stock_price, *d_option_strike, *d_option_years,
      *d_pt_weight, *d_pt_contrib;

  // Allocate CPU memory for buffers
  call_price = (float *)malloc(OPT_SZ);
  stock_price = (float *)malloc(OPT_SZ);
  option_strike = (float *)malloc(OPT_SZ);
  option_years = (float *)malloc(OPT_SZ);
  pt_weight = (float *)malloc(OPT_SZ);

  // Allocate GPU memory for buffers
  cudaMalloc((void **)&d_call_price, OPT_SZ);
  cudaMalloc((void **)&d_stock_price, OPT_SZ);
  cudaMalloc((void **)&d_option_strike, OPT_SZ);
  cudaMalloc((void **)&d_option_years, OPT_SZ);
  cudaMalloc((void **)&d_pt_weight, OPT_SZ);
  cudaMalloc((void **)&d_pt_contrib, OPT_SZ);

  srand(5347);

  // Generate options set

  for (int i = 0; i < OPT_N; i++) {
    call_price[i] = 0.0f;
    stock_price[i] = RandFloat(5.0f, 30.0f);
    option_strike[i] = RandFloat(1.0f, 100.0f);
    option_years[i] = RandFloat(0.25f, 10.0f);
    pt_weight[i] = RandFloat(-2.0f, 2.0f);
  }

  // Copy options data to GPU memory for further processing
  cudaMemcpy(d_stock_price, stock_price, OPT_SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(d_option_strike, option_strike, OPT_SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(d_option_years, option_years, OPT_SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pt_weight, pt_weight, OPT_SZ, cudaMemcpyHostToDevice);

  int numThreads = 256;
  int numBlocksBS = (OPT_N + numThreads - 1) / numThreads;

  if (numBlocksBS > 65535)
    numBlocksBS = 65535;


  cudaEventCreate(&start_timer);
  cudaEventCreate(&stop_timer);
  cudaEventRecord(start_timer, 0);

  black_scholes<<<numBlocksBS, numThreads>>>(d_call_price, d_stock_price,
                                             d_option_strike, d_option_years,
                                             RISKFREE, VOLATILITY, OPT_N);

  // CUDA_CHECK(cudaGetLastError());

  weightMultiply<<<numBlocksBS, numThreads>>>(d_call_price, d_pt_weight,
                                              d_pt_contrib, OPT_N);

  const int blocksReduce = 1024;
  float *d_redA, *d_redB;
  CUDA_CHECK(
      cudaMalloc((void **)&d_redA, (size_t)blocksReduce * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_redB, (size_t)blocksReduce * sizeof(float)));

  float *d_one = reduce_to_one_gpu(d_pt_contrib, OPT_N, numThreads,
                                   blocksReduce, d_redA, d_redB);

  // cudaDeviceSynchronize();
  cudaStreamSynchronize(0);

  float portfolio_value = 0.0;
  CUDA_CHECK(cudaMemcpy(&portfolio_value, d_one, sizeof(float),
                        cudaMemcpyDeviceToHost));

  cudaEventRecord(stop_timer, 0);
  cudaEventElapsedTime(&elapsed_time, start_timer, stop_timer); 

  printf("The total value of the call options portfolio is: %f\n",
         portfolio_value);
  printf("CUDA elapsed execution %f/ms\n", elapsed_time);

  free(call_price);
  free(stock_price);
  free(option_strike);
  free(option_years);
  free(pt_weight);

  CUDA_CHECK(cudaFree(d_call_price));
  CUDA_CHECK(cudaFree(d_stock_price));
  CUDA_CHECK(cudaFree(d_option_strike));
  CUDA_CHECK(cudaFree(d_option_years));
  CUDA_CHECK(cudaFree(d_pt_weight));
  CUDA_CHECK(cudaFree(d_pt_contrib));
  CUDA_CHECK(cudaFree(d_redA));
  CUDA_CHECK(cudaFree(d_redB));

  cudaEventDestroy(start_timer);
  cudaEventDestroy(stop_timer);

  return 0;
}
