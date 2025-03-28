import torch
import ctypes
from torch.utils.cpp_extension import load_inline
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

"""
def load_custom_ptx_kernel(ptx_file_path):
    # Read the PTX file
    with open(ptx_file_path, 'r') as f:
        ptx_code = f.read()
    
    # Load the PTX module using PyTorch's CUDA interface
    # cuda_mod = torch.cuda.CUDAModule(ptx_code)
    cuda_ext = load_inline(
        name="custom_clamp_ext",
        cpp_sources="",  # No C++ code needed
        cuda_sources="extern \"C\" __global__ void dummy_kernel() {}",  # Dummy kernel
        functions=["triton_"],  # Expose the PTX function
        extra_cuda_cflags=[],
        with_cuda=True,
        code=ptx_code  # Include our PTX code
        )
    
    # Get the function from the module (use the name as defined in your PTX)
    #kernel_function = cuda_mod.get_function("triton_")
    
    return getattr(cuda_ext, "triton_")

"""

def load_custom_ptx_kernel(ptx_file_path):
    with open(ptx_file_path, 'r') as f:
        ptx_code = f.read()
    mod = cuda.module_from_buffer(ptx_code.encode())
    return mod.get_function("triton_")

def test_modified_ptx():
    # Input data
    input_tensor = torch.randn(512, device='cuda')
    output_tensor = torch.zeros_like(input_tensor)
    
    # Create a debug buffer to store diagnostic values
    debug_buffer = torch.zeros(10, dtype=torch.int32, device='cuda')
    
    # Load and run your kernel
    kernel = load_custom_ptx_kernel("clamp_kernel_modified.ptx")
    kernel(
        grid=(2, 1, 1),
        block=(256, 1, 1),
        args=[
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            ctypes.c_int(input_tensor.numel()),
            debug_buffer.data_ptr()  # Add this as an extra parameter
        ]
    )
    
    # Check the debug buffer values
    print("Debug values:", debug_buffer.cpu().numpy())
    
    return output_tensor

if __name__ == "__main__":
    # Create a sample tensor
    input_tensor = torch.randn(512, device='cuda')
    output_tensor = torch.zeros_like(input_tensor)
    
    # Run the custom clamping
    result = test_modified_ptx()
    #result = run_custom_clamp(input_tensor, output_tensor)
    print(result)
