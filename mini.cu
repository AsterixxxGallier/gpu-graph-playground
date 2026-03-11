#include <stdio.h>

__global__ void testKernel()
{
    printf("Hello from GPU\n");
}

int main()
{
    testKernel<<<1,1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    printf("Hi!\n");
}