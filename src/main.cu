#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <numeric>

#include "common/helper_cuda.hpp"
#include "init.cuh"
#include "one_sweep.cuh"

constexpr auto radix = 256;  // fixed for 32-bit unsigned int
constexpr auto radixPasses = 4;

[[nodiscard]] constexpr int globalHistThreadblocks(const int size) {
  return 2048;
}

[[nodiscard]] constexpr int binningThreadblocks(const int size) {
  // looks like we want to process 15 items per thread
  // and since 512 threads was used, we have
  constexpr auto partitionSize = 7680;
  return size / partitionSize;
}

constexpr auto laneCount = 32;       // fixed for NVIDIA GPUs
constexpr auto globalHistWarps = 8;  // configurable
constexpr auto digitBinWarps = 16;   // configurable

// 8x32=256 threads
const dim3 globalHistDim(laneCount, globalHistWarps, 1);

// 16x32=512 threads
const dim3 digitBinDim(laneCount, digitBinWarps, 1);

struct RadixSortData {
  explicit RadixSortData(const int n) : n(n) {
    checkCudaErrors(cudaMallocManaged(&u_sort, n * sizeof(unsigned int)));
    checkCudaErrors(cudaMallocManaged(&u_sort_alt, n * sizeof(unsigned int)));
    checkCudaErrors(
        cudaMallocManaged(&u_index, radixPasses * sizeof(unsigned int)));
    checkCudaErrors(cudaMallocManaged(
        &u_global_histogram, radix * radixPasses * sizeof(unsigned int)));
    for (auto& pass_histogram : u_pass_histogram) {
      checkCudaErrors(cudaMallocManaged(
          &pass_histogram,
          radix * binningThreadblocks(n) * sizeof(unsigned int)));
    }
  }

  ~RadixSortData() {
    checkCudaErrors(cudaFree(u_sort));
    checkCudaErrors(cudaFree(u_sort_alt));
    checkCudaErrors(cudaFree(u_index));
    checkCudaErrors(cudaFree(u_global_histogram));
    for (auto& pass_histogram : u_pass_histogram) {
      checkCudaErrors(cudaFree(pass_histogram));
    }
  }

  [[nodiscard]] bool IsSorted(const int n) const {
    return std::is_sorted(u_sort, u_sort + n);
  }

  void InitRandom(const int seed) {
    constexpr auto block_size = 768;
    const auto num_blocks = (n + block_size - 1) / block_size;
    k_InitRandom<<<num_blocks, block_size>>>(u_sort, n, seed);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void DsipatchGlobalHistogram() {
    k_GlobalHistogram<<<globalHistThreadblocks(n), globalHistDim>>>(
        u_sort, u_global_histogram, n);
  }

  void DispatchDigitBinning(const int pass) {
    if (pass % 2 == 0) {
      k_DigitBinning<<<binningThreadblocks(n), digitBinDim>>>(
          u_global_histogram,
          u_sort,
          u_sort_alt,
          u_pass_histogram[pass],
          u_index,
          n,
          pass * 8);
    } else {
      k_DigitBinning<<<binningThreadblocks(n), digitBinDim>>>(
          u_global_histogram,
          u_sort_alt,
          u_sort,
          u_pass_histogram[pass],
          u_index,
          n,
          pass * 8);
    }
  }

  int n;
  unsigned int* u_sort;
  unsigned int* u_sort_alt;
  unsigned int* u_index;
  unsigned int* u_global_histogram;
  std::array<unsigned int*, radixPasses> u_pass_histogram;
};

int main(const int argc, const char* argv[]) {
  constexpr int sizeExponent = 28;
  int n = 1 << sizeExponent;  // 256M elements

  if (argc > 1) {
    n = std::atoi(argv[1]);
  }

  // check n > 8096, and n is smaller than 2^28
  if (n < 8096 || n > (1 << 28)) {
    std::cerr << "n must be between 8096 and 2^28\n";
    return 1;
  }

  std::cout << "n = " << n << '\n';

  auto data_ptr = std::make_unique<RadixSortData>(n);

  std::cout << "initializing...\n";
  constexpr auto seed = 114514;
  data_ptr->InitRandom(seed);

  auto result = data_ptr->IsSorted(n);
  std::cout << "Before sorting: Is sorted ? " << std::boolalpha << result
            << '\n';

  std::cout << "start sorting...\n";

  data_ptr->DsipatchGlobalHistogram();
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