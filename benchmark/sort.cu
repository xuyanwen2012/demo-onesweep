
#include <benchmark/benchmark.h>

#include "init.cuh"
#include "one_sweep.cuh"
#include "sync.h"

namespace bm = benchmark;

constexpr auto kRadix = 256;  // fixed for 32-bit unsigned int
constexpr auto kRadixPasses = 4;

static void BM_RadixSortHistogram(bm::State &st) {
  constexpr auto num_items = 10'000'000;
  const auto num_blocks = st.range(0);

  unsigned int *d_data;
  BENCH_CUDA_TRY(cudaMallocManaged(&d_data, num_items * sizeof(unsigned int)));

  unsigned int *d_histogram;
  BENCH_CUDA_TRY(cudaMallocManaged(
      &d_histogram,
      static_cast<size_t>(kRadix) * kRadixPasses * sizeof(unsigned int)));

  //   // init random data
  //   {
  //     constexpr auto threads = 768;
  //     constexpr auto blocks = (num_items + threads - 1) / threads;
  //     k_InitRandom<<<blocks, threads>>>(d_data, num_items, 114514 /*seeds*/);
  //     BENCH_CUDA_TRY(cudaDeviceSynchronize());
  //   }

  int a, b, c;

  for (auto _ : st) {
    cuda_event_timer raii{st, true};
    constexpr auto threads = 768;
    // k_GlobalHistogram<<<num_blocks, threads>>>(d_data, d_histogram,
    // num_items);
    c = a + b;
    bm::DoNotOptimize(c);
  }

  BENCH_CUDA_TRY(cudaFree(d_data));
  BENCH_CUDA_TRY(cudaFree(d_histogram));
}

BENCHMARK(BM_RadixSortHistogram)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
