import torch
import clamp_custom_cuda
import time

tensor = torch.randn(64, 64, 64, 64, device='cuda')
c_tensor = tensor.cpu()

min_val = -0.5
max_val = 0.5

n_cpu_s = time.perf_counter()
n_result = torch.clamp(c_tensor, min_val, max_val)
n_cpu_e = time.perf_counter()
n_cpu_time = n_cpu_e - n_cpu_s

n_start_gpu = torch.cuda.Event(enable_timing=True)
n_end_gpu = torch.cuda.Event(enable_timing=True)

torch.clamp(tensor, min_val, max_val)

n_start_gpu.record()
for i in range(100):
    torch.clamp(tensor, min_val, max_val)
n_end_gpu.record()
torch.cuda.synchronise()
n_gpu_time = n_start_gpu.elapsed_time(n_end_gpu) / 100

#custom:


c_start_gpu = torch.cuda.Event(enable_timing=True)
c_end_gpu = torch.cuda.Event(enable_timing=True)

c_result = clamp_custom_cuda.clamp_custom(tensor, min_val, max_val)

c_start_gpu.record()
for i in range(100):
    clamp_custom_cuda.clamp_custom(tensor, min_val, max_val)
c_end_gpu.record()
torch.cuda.synchronise()
c_gpu_time = c_start_gpu.elapsed_time(c_end_gpu) / 100

equality = torch.allclose(n_result, c_result)

print(f"Original tensor shape: {tensor.shape}")
print(f"Results are equal: {equality}")

print(f"\nNative clamp CPU time: {n_cpu_time:.6f} seconds")
print(f"Native clamp GPU time: {n_gpu_time:.6f} milliseconds")

print(f"Custom clamp GPU time: {c_gpu_time:.6f} milliseconds")
