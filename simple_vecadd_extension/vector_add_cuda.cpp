#include <torch/extension.h>
#include <iostream>
#include <vector>

torch::Tensor vector_add_cuda(torch::Tensor vecA, torch::Tensor vecB);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor vector_add(torch::Tensor vecA, torch::Tensor vecB) {
    CHECK_INPUT(vecA);
    CHECK_INPUT(vecB);

    return vector_add_cuda(vecA, vecB);
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "Add two Vectors (CUDA)");
}
