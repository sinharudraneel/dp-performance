#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

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
    //if (threadIdx.x == 0 && blockIdx.x == 0) {
    //	printf("ndim: %lld\n", ndim);
    //	printf("")
    //}
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

    int64_t *d_sizes, *d_strides;
    cudaMalloc(&d_sizes, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_sizes, sizes, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    //std::vector<int64_t> h_sizes(ndim), h_strides(ndim);
    //cudaMemcpy(h_sizes.data(), d_sizes, ndim * sizeof(int64_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_strides.data(), d_strides, ndim * sizeof(int64_t), cudaMemcpyDeviceToHost);
    
    //std::cout << "Sizes and strides on GPU:" << std::endl;
    //for (int i = 0; i < ndim; ++i) {
    //    std::cout << "sizes[" << i << "]: " << h_sizes[i] 
    //              << ", strides[" << i << "]: " << h_strides[i] << std::endl;
    //}

    clamp_custom_kernel<<<blocks, threads>>>(
        input, output, d_sizes, d_strides, numel, ndim, min_val, max_val);

    cudaFree(d_sizes);
    cudaFree(d_strides);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    	throw std::runtime_error(cudaGetErrorString(err));
    }
}

extern "C" void clamp_custom_cuda(
	double* input,
	double* output,
	const int64_t* sizes,
	const int64_t* strides,
	int64_t	numel,
	int ndim,
	double min_val,
	double max_val	
) {
	launch_clamp_custom_kernel<double>(input, output, sizes, strides, numel, ndim, min_val, max_val);
}
