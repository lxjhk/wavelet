#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  compute capability major, minor: %d %d\n", prop.major, prop.minor);
    printf("  totalGlobalMem: %zu\n", prop.totalGlobalMem);
    printf("  regsPerBlock: %d\n", prop.regsPerBlock);
    printf("  sharedMemPerBlock: %zu\n", prop.sharedMemPerBlock);
    printf("  warpSize: %d\n", prop.warpSize);
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("  maxThreadsDim %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  maxGridSize %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
