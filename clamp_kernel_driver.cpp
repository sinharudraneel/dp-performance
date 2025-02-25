#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare the kernel function that is in the shared object
extern "C" void clamp_custom_cuda(double* input, double* output, int* sizes, int* strides, int64_t numel, int ndim, double min_val, double max_val);

int main() {
    // Define tensor dimensions and data
    int ndim = 3;
    int64_t numel = 8;
    double min_val = 0.0;
    double max_val = 1.0;

    // Example input tensor (flattened as a 1D array)
    std::vector<double> input = { 0.2, 0.8, 1.2, 1.5, -0.1, 0.3, 0.9, 1.1 };
    std::vector<double> output(numel, 0);

    // Sizes and strides (simplified for 1D case here)
    std::vector<int64_t> sizes = { numel };
    std::vector<int64_t> strides = { 1 };

    // Allocate device memory
    double* d_input;
    double* d_output;
    int* d_sizes;
    int* d_strides;

    cudaMalloc(&d_input, input.size() * sizeof(double));
    cudaMalloc(&d_output, output.size() * sizeof(double));
    cudaMalloc(&d_sizes, sizes.size() * sizeof(int64_t));
    cudaMalloc(&d_strides, strides.size() * sizeof(int64_t));

    // Copy input data to device memory
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, sizes.data(), sizes.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, strides.data(), strides.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel (this assumes the kernel has been compiled into a shared object)
    clamp_custom_cuda(d_input, d_output, d_sizes, d_strides, numel, ndim, min_val, max_val);

    // Copy the output data from device to host
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the output
    std::cout << "Clamped Output: ";
    for (double val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Free allocated device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_sizes);
    cudaFree(d_strides);

    return 0;
}

