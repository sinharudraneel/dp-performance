#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor clamp_custom_cuda(
    torch::Tensor input, 
    torch::Scalar min_val, 
    torch::Scalar max_val
    );

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor clamp_custom(
    torch::Tensor input,
    double min_val, 
    double max_val
    ) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    at::cuda::CUDAGuard device_guard(input.device());

    return clamp_custom_cuda(input, min_val, max_val);
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("clamp_custom", &clamp_custom, "Clamp a tensor (CUDA)");
}
