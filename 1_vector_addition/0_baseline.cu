#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < N) c[tid] = a[tid] + b[tid];
}

int main() {
	constexpr int N = 1 << 16;
	constexpr size_t bytes = sizeof(int) * N;

	int *c_a, *c_b, *c_c;

	c_a = (int*) malloc(bytes);
	c_b = (int*) malloc(bytes);
	c_c = (int*) malloc(bytes);

	// Initialize random numbers in each array
	for (int i = 0; i < N; i++) {
		c_a[i] = rand() % 100;
		c_b[i] = rand() % 100;
	}

	// Allocate memory on the device
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Copy data from the host to the device (CPU -> GPU)
	cudaMemcpy(d_a, c_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, c_b, bytes, cudaMemcpyHostToDevice);

	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS; // ceil(N / NUM_THREADS)

	// Launch the kernel on the GPU (async)
	vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

	// this acts as a synchronization barrier followed by a memcpy (GPU -> CPU)
	cudaMemcpy(c_c, d_c, bytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		assert(c_c[i] == c_a[i] + c_b[i]);
	}
	free(c_a);
	free(c_b);
	free(c_c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "COMPLETED SUCCESSFULLY\n";
	return 0;
}