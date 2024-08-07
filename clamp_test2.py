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

tensor_2d = torch.tensor([[-2, 0, 3], [5, 8, -1], [4, 6, 2]], dtype=torch.float32)
clamped_2d = torch.clamp(tensor_2d, min=0, max=5)
print("\n2D clamped tensor (torch.clamp):")
print(clamped_2d)

tensor_2d_cuda = tensor_2d.cuda()
clamped_custom_2d = clamp_custom_cuda.clamp_custom(tensor_2d_cuda, min_val, max_val)
torch.cuda.synchronize()
clamped_custom_2d_cpu = clamped_custom_2d.cpu()
print("2D clamped tensor (custom_clamp):")
print(clamped_custom_2d_cpu)


assert torch.allclose(clamped_1d, clamped_custom_1d_cpu), "1D results do not match"
assert torch.allclose(clamped_2d, clamped_custom_2d_cpu), "2D results do not match"
print("\nAll results match!")
