#include <stdio.h>
#include <cassert>
#include <iostream>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (tid < N) c[tid] = a[tid] + b[tid];
}

int main() {
  // Array size of 2^16 (65536 elements)
  constexpr int N = 1 << 16;
  constexpr size_t bytes = sizeof(int) * N;

  int *a, *b, *c;

  // Allocating memory for these pointers (automatic management with unified memory over CPU and GPU)
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);
  
  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }
  
  int BLOCK_SIZE = 1 << 10;
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // ceil(N / BLOCK_SIZE)

  vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

  cudaDeviceSynchronize();
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }
  cudaFree(a);
	cudaFree(b);
  cudaFree(c);

  std::cout << "COMPLETED SUCCESSFULLY!\n";
  return 0;
}