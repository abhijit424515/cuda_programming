#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::vector;

// Pull out matrix and shared memory tile size 
const int SHMEM_SIZE = 1 << 10;

__global__ void matrixMul(const int *a, const int *b, int *c, int N, int tile_size) {
	// Statically allocated shared memory
	__shared__ int s_a[SHMEM_SIZE];
	__shared__ int s_b[SHMEM_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int row = by * tile_size + ty;
	int col = bx * tile_size + tx;

	int temp = 0;
	for (int i=0; i < (N/tile_size); i++) {
		s_a[ty * tile_size + tx] = a[row*N + i*tile_size + tx];
		s_b[ty * tile_size + tx] = b[i*tile_size*N + ty*N + col];
	
		__syncthreads();
		for (int j=0; j<tile_size; j++) {
			temp += s_a[ty*tile_size + j] * s_b[j*tile_size + tx];
		}
		__syncthreads();
	}

	c[row*N + col] = temp;
}

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int tmp = 0;
			for (int k = 0; k < N; k++) {
			tmp += a[i * N + k] * b[k * N + j];
			}
			assert(tmp == c[i * N + j]);
		}
	}
}

int main() {
	const int N = 1 << 10;
	constexpr size_t bytes = N * N * sizeof(int);

	// Host vectors
	vector<int> h_a(N * N);
	vector<int> h_b(N * N);
	vector<int> h_c(N * N);

	// Initialize matrices
	std::generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
	std::generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

	// Allocate device memory
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Copy data to the device
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	int THREADS = 32;
	int BLOCKS = N / THREADS;

	// Use dim3 structs for block  and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	// Launch kernel
	matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N, THREADS);

	cudaDeviceSynchronize();
	std::cout << "GPU MATMUL FINISHED\n";

	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
	verify_result(h_a, h_b, h_c, N);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "COMPLETED SUCCESSFULLY\n";
	return 0;
}