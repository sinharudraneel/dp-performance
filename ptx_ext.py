import torch
import os


def extract_ptx(func, *args, **kwargs):
    args = [arg.cuda() if isinstance(arg, torch.Tensor) else arg for arg in args]
    kwargs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k in kwargs.items()}


    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
            ) as prof:
        result = func(*args, **kwargs)
        torch.cuda.synchronize()

    print("\nProfile Output")
    print(prof)

    return result


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available");
    torch.cuda.set_device(0);

    x = torch.randn(1000000, device=0)
    lower = -0.5
    upper = 0.5

    result = extract_ptx(torch.clamp, x, lower, upper)
    print(result)

