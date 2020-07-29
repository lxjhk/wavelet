#include <cuda.h>
#include <iostream>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
using namespace std;

template <typename scalar_t>
__global__ void AddInts(scalar_t *a, scalar_t *b, int count){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < count){
        a[id] += b[id];
    }
}

std::vector<torch::Tensor> add_test(
    torch::Tensor a,
    torch::Tensor b) {
  const int threads = 256;
  const int state_size = a.size(0);

  AT_DISPATCH_ALL_TYPES(a.type(), "AddInts", ([&] {
    AddInts<scalar_t><<<1, threads>>>(
        a.data<scalar_t>(),
        b.data<scalar_t>(),
        state_size);
  }));

  return {a};
}

void enablePeerAccess() {
    std::cout << "Enabling Peer Access between GPU 2, 3." << std::endl;
    cudaSetDevice(2);
    cudaDeviceEnablePeerAccess(2,0);
    cudaSetDevice(3);
    cudaDeviceEnablePeerAccess(3,0);
    cudaDeviceSynchronize();
}
void cudaSync() {
    cudaSetDevice(2);
    cudaDeviceSynchronize();
    cudaSetDevice(3);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(cuda_p2p_kenan, m) {
  m.def("enablePeerAccess", &enablePeerAccess, "CUDA enablePeerAccess");
  m.def("add_test", &add_test, "CUDA add_test");
  m.def("cudaSync", &cudaSync, "CUDA cudaSync");
}