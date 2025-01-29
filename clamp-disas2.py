import torch
import time
import os
import tempfile
from pathlib import Path

def find_and_save_ptx(name="clamp_kernel"):
    """Find and save the most recently generated PTX file"""
    try:
        # Check common torch cache locations
        cache_paths = [
            Path.home() / '.cache' / 'torch_extensions',
            Path(tempfile.gettempdir()) / 'torch_extensions',
            Path.home() / '.cache' / 'torch_compile_debug',
            Path('/tmp/torchinductor_root'),  # Common Linux location
            Path.cwd() / 'torch_compile_debug',
        ]
        
        print("\nSearching for PTX files in:")
        for path in cache_paths:
            print(f"- {path}")
        
        newest_ptx = None
        newest_time = 0
        
        # Search for PTX files
        found_files = []
        for cache_path in cache_paths:
            if not cache_path.exists():
                continue
                
            print(f"\nSearching in {cache_path}")
            for root, _, files in os.walk(cache_path):
                for file in files:
                    if file.endswith('.ptx'):
                        file_path = Path(root) / file
                        found_files.append((file_path, file_path.stat().st_mtime))
                        print(f"Found PTX: {file_path}")
                        if file_path.stat().st_mtime > newest_time:
                            newest_time = file_path.stat().st_mtime
                            newest_ptx = file_path

        if not found_files:
            print("\nNo PTX files found in any location")
            return
            
        print(f"\nFound {len(found_files)} PTX files in total")
        print(f"Most recent PTX: {newest_ptx}")
            
        # Read and save to current directory
        output_file = Path.cwd() / f"{name}.ptx"
        with open(newest_ptx, 'r') as src, open(output_file, 'w') as dst:
            ptx_content = src.read()
            dst.write(ptx_content)
            
        print(f"\nPTX file saved to: {output_file.absolute()}")
        print("PTX content preview:")
        print("=" * 40)
        print(ptx_content[:500] + "...")
        print("=" * 40)
        
    except Exception as e:
        print(f"Error saving PTX file: {e}")
        import traceback
        traceback.print_exc()

def run_clamp_demo(tensor_size=(10000, 10000), min_val=-1.0, max_val=1.0):
    # Print current working directory
    print(f"Current working directory: {Path.cwd().absolute()}")
    
    # Force CUDA - will raise error if not available
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA GPU")
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Enable TensorFloat-32 (TF32) on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create and compile the clamp function using torch.compile
    @torch.compile(backend="inductor", mode="max-autotune", fullgraph=True)
    def clamp_tensor(x):
        return torch.clamp(x, min=min_val, max=max_val)
    
    # Create a large random tensor directly on GPU
    print(f"\nGenerating random tensor of size {tensor_size} on GPU...")
    x = torch.randn(tensor_size, device=device, dtype=torch.float32)
    
    # Ensure tensor is on GPU
    assert x.is_cuda, "Tensor not on GPU!"
    
    print(f"Initial tensor stats:")
    print(f"Min: {x.min().item():.4f}")
    print(f"Max: {x.max().item():.4f}")
    print(f"Mean: {x.mean().item():.4f}")
    
    # Warmup run for compilation
    print("\nPerforming warmup compilation...")
    warmup = clamp_tensor(torch.randn((100, 100), device=device))
    torch.cuda.synchronize()
    
    # Try to find and save PTX
    find_and_save_ptx()
    
    # Time the clamp operation
    print(f"\nClamping values between {min_val} and {max_val}...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    clamped_x = clamp_tensor(x)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"\nOperation completed in {end_time - start_time:.4f} seconds")
    
    return clamped_x

if __name__ == "__main__":
    # Set environment variables for debugging
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    
    # Make sure the torch_compile_debug directory exists in current directory
    debug_dir = Path.cwd() / 'torch_compile_debug'
    debug_dir.mkdir(exist_ok=True)
    
    result = run_clamp_demo()
