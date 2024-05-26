#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
    return 0;
}
