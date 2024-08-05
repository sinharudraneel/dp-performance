import torch
import clamp_custom_cuda
import numpy as np

assert torch.cuda.is_available(), "CUDA is not available. This script requires a GPU."

tensor_1d = torch.tensor([-2, 0, 3, 5, 8], dtype=torch.float32)
clamped_1d = torch.clamp(tensor_1d, min=0, max=5)
print("1D clamped tensor (torch.clamp)")
print(clamped_1d)

min_val = 0.0
max_val = 5.0
tensor_1d_cuda = tensor_1d.cuda()
clamped_custom_1d = clamp_custom_cuda.clamp_custom(tensor_1d_cuda, min_val, max_val)
torch.cuda.synchronize()
clamped_custom_1d_cpu = clamped_custom_1d.cpu()
print("1d clamped tensor (custom_clamp)")
print(clamped_custom_1d_cpu)
