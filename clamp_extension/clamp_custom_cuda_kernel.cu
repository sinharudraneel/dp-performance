#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void clamp_custom_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides,
    int64_t numel,
    int ndim,
    scalar_t min_val,
    scalar_t max_val
) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        int64_t input_index = 0;
        int64_t remaining = index;
        for (int i = 0; i < ndim; ++i) {
            int64_t coord = remaining / strides[i];
            remaining %= strides[i];
            input_index += coord * strides[i];
        }
        scalar_t val = input[input_index];
        output[input_index] = max(min(val, max_val), min_val);
    }
}

template <typename scalar_t>
void launch_clamp_custom_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int64_t* sizes,
    const int64_t* strides,
    int64_t numel,
    int ndim,
    scalar_t min_val,
    scalar_t max_val
) {
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;
    
    clamp_custom_kernel<<<blocks, threads>>>(
        input, output, sizes, strides, numel, ndim, min_val, max_val);
}

torch::Tensor clamp_custom_cuda(
    torch::Tensor input,
    double min_val,
    double max_val
) {
    auto output = torch::empty_like(input);
    auto sizes = input.sizes().vec();
    auto strides = input.strides().vec();

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "clamp_custom_cuda", ([&] {
        launch_clamp_custom_kernel<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            sizes.data(),
            strides.data(),
            input.numel(),
            input.dim(),
            static_cast<scalar_t>(min_val),
            static_cast<scalar_t>(max_val)
        );
    }));
    return output;
}
