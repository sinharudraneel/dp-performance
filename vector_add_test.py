import torch
import vector_add__cuda
from torch.profiler import profile, record_function, ProfilerActivity

a = torch.rand(1000000, device='cuda')
b = torch.rand(1000000, device='cuda')

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True,
             profile_memory=True,
             with_stack=True) as prof:
    with record_function("vector_add_operation"):
        c = vector_add__cuda.vector_add(a,b)
    print(c)


print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
