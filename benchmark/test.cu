
#include <benchmark/benchmark.h>

namespace bm = benchmark;

constexpr size_t f32s_in_cacheline_k = 64 / sizeof(float);
constexpr size_t f32s_in_halfline_k = f32s_in_cacheline_k / 2;

struct alignas(64) f32_array_t {
  float raw[f32s_in_cacheline_k * 2];
};

static void f32_pairwise_accumulation(bm::State &state) {
  f32_array_t a, b, c;
  for (auto _ : state)
    for (size_t i = f32s_in_halfline_k; i != f32s_in_halfline_k * 3; ++i)
      bm::DoNotOptimize(c.raw[i] = a.raw[i] + b.raw[i]);
}

static void f32_pairwise_accumulation_aligned(bm::State &state) {
  f32_array_t a, b, c;
  for (auto _ : state)
    for (size_t i = 0; i != f32s_in_halfline_k; ++i)
      bm::DoNotOptimize(c.raw[i] = a.raw[i] + b.raw[i]);
}

BENCHMARK(f32_pairwise_accumulation);

BENCHMARK(f32_pairwise_accumulation_aligned);

BENCHMARK_MAIN();
