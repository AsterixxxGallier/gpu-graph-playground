//#include <cuda_runtime.h>
//#include <stdio.h>
//
//int main() {
////    unsigned long long hostCount = 0;
////    unsigned long long* deviceCount;
////    cudaMalloc(&deviceCount, sizeof(unsigned long long));
////    cudaMemset(deviceCount, 100, sizeof(unsigned long long));
////
////    cudaMemcpy(&hostCount, deviceCount,
////               sizeof(unsigned long long),
////               cudaMemcpyDeviceToHost);
////    cudaFree(deviceCount);
////
////    printf("count: %llu\n", hostCount);
////
////    return 0;
//
//    unsigned long long hostCount = 0;
//    unsigned long long* deviceCount;
//
//    cudaMalloc(&deviceCount, sizeof(unsigned long long));
//
//    unsigned long long value = 100;
//    cudaMemcpy(deviceCount, &value, sizeof(value), cudaMemcpyHostToDevice);
//
//    cudaMemcpy(&hostCount, deviceCount,
//               sizeof(unsigned long long),
//               cudaMemcpyDeviceToHost);
//
//    cudaFree(deviceCount);
//
//    printf("count: %llu\n", hostCount);
//
//    return 0;
//}

#include <cuda_runtime.h>
#include <stdio.h>

__global__
void setToOneHundred(unsigned long long *count) {
    *count = 12341245521412451;
}

int main() {
    unsigned long long hostCount = 0;
    unsigned long long *deviceCount;

    cudaError_t err;

    err = cudaMalloc(&deviceCount, 8);
    printf("cudaMalloc: %s\n", cudaGetErrorString(err));

    setToOneHundred<<<1, 1>>>(deviceCount);

//    unsigned long long value = 12341245521412451;
//
//    err = cudaMemcpy(deviceCount, &value, 8, cudaMemcpyHostToDevice);
//    printf("Memcpy H->D: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();

    cudaMemcpy(&hostCount, deviceCount, 8, cudaMemcpyDeviceToHost);

    cudaFree(deviceCount);

    printf("count: %llu\n", hostCount);
}