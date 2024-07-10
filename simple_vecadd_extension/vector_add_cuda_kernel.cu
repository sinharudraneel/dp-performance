#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void vector_add_kernel (
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int64_t size
);

torch::Tensor vector_add_cuda (torch::Tensor a, torch::Tensor b) {
    if (a.sizes() != b.sizes()) {
        throw std::runtime_error("Input tensors must have the same shape");
    }

    auto c = torch::zeros_like(a);

    const int64_t size = a.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(a.type(), "add_vectors_cuda", ([&] {
        vector_add_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            size
        );
    }));

    return c;

}

template <typename scalar_t>
__global__ void vector_add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, 
    scalar_t* __restrict__ c,
    int64_t size) {
        
        int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            c[index] = a[index] + b[index];
        }
    }
