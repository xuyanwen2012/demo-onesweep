#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <numeric>

#include "common/helper_cuda.hpp"
#include "init.cuh"
#include "one_sweep.cuh"

constexpr auto kRadix = 256;  // fixed for 32-bit unsigned int
constexpr auto kRadixPasses = 4;

[[nodiscard]] constexpr int GlobalHistThreadBlocks(const int size) {
  return 1;  // was 2048
  // return 2048;
}

[[nodiscard]] constexpr int BinningThreadBlocks(const int size) {
  // looks like we want to process 15 items per thread
  // and since 512 threads was used, we have
  constexpr auto partition_size = 7680;
  return size / partition_size;
}

constexpr auto kLaneCount = 32;       // fixed for NVIDIA GPUs
constexpr auto kGlobalHistWarps = 8;  // configurable
constexpr auto kDigitBinWarps = 16;   // configurable

// 8x32=256 threads
const dim3 kGlobalHistDim(kLaneCount, kGlobalHistWarps, 1);

// 16x32=512 threads
const dim3 kDigitBinDim(kLaneCount, kDigitBinWarps, 1);

struct RadixSortData {
  explicit RadixSortData(const int n) : n(n) {
    checkCudaErrors(cudaMallocManaged(&u_sort, n * sizeof(unsigned int)));
    checkCudaErrors(cudaMallocManaged(&u_sort_alt, n * sizeof(unsigned int)));
    checkCudaErrors(
        cudaMallocManaged(&u_index, kRadixPasses * sizeof(unsigned int)));
    checkCudaErrors(cudaMallocManaged(
        &u_global_histogram, kRadix * kRadixPasses * sizeof(unsigned int)));
    for (auto& pass_histogram : u_pass_histogram) {
      checkCudaErrors(cudaMallocManaged(
          &pass_histogram,
          kRadix * BinningThreadBlocks(n) * sizeof(unsigned int)));
    }
  }

  ~RadixSortData() {
    checkCudaErrors(cudaFree(u_sort));
    checkCudaErrors(cudaFree(u_sort_alt));
    checkCudaErrors(cudaFree(u_index));
    checkCudaErrors(cudaFree(u_global_histogram));
    for (const auto& pass_histogram : u_pass_histogram) {
      checkCudaErrors(cudaFree(pass_histogram));
    }
  }

  [[nodiscard]] bool IsSorted(const int n) const {
    return std::is_sorted(u_sort, u_sort + n);
  }

  void InitRandom(const int seed) const {
    constexpr auto block_size = 768;
    const auto num_blocks = (n + block_size - 1) / block_size;
    k_InitRandom<<<num_blocks, block_size>>>(u_sort, n, seed);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void DispatchGlobalHistogram() const {
    k_GlobalHistogram<<<GlobalHistThreadBlocks(n), kGlobalHistDim>>>(
        u_sort, u_global_histogram, n);
  }

  void DispatchDigitBinning(const int pass) const {
    unsigned int* input;
    unsigned int* output;

    if (pass % 2 == 0) {
      input = u_sort;
      output = u_sort_alt;
    } else {
      input = u_sort_alt;
      output = u_sort;
    }

    k_DigitBinning<<<BinningThreadBlocks(n), kDigitBinDim>>>(
        u_global_histogram,
        input,
        output,
        u_pass_histogram[pass],
        u_index,
        n,
        pass * 8);
  }

  int n;
  unsigned int* u_sort;
  unsigned int* u_sort_alt;
  unsigned int* u_index;
  unsigned int* u_global_histogram;
  std::array<unsigned int*, kRadixPasses> u_pass_histogram;
};

int main(const int argc, const char* argv[]) {
  constexpr int size_exponent = 28;
  int n = 1 << size_exponent;  // 256M elements

  if (argc > 1) {
    n = std::strtol(argv[1], nullptr, 10);
  }

  // check n > 8096, and n is smaller than 2^28
  if (n < 8096 || n > (1 << 28)) {
    std::cerr << "n must be between 8096 and 2^28\n";
    return 1;
  }

  std::cout << "n = " << n << '\n';

  const auto data_ptr = std::make_unique<RadixSortData>(n);

  std::cout << "initializing...\n";
  constexpr auto seed = 114514;
  data_ptr->InitRandom(seed);

  auto result = data_ptr->IsSorted(n);
  std::cout << "Before sorting: Is sorted ? " << std::boolalpha << result
            << '\n';

  std::cout << "start sorting...\n";

  data_ptr->DispatchGlobalHistogram();
  data_ptr->DispatchDigitBinning(0);
  data_ptr->DispatchDigitBinning(1);
  data_ptr->DispatchDigitBinning(2);
  data_ptr->DispatchDigitBinning(3);

  checkCudaErrors(cudaDeviceSynchronize());

  result = data_ptr->IsSorted(n);
  std::cout << "After sorting: Is sorted ? " << std::boolalpha << result
            << '\n';

  return 0;
}
