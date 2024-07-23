import torch

def main():
    # Set the device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a tensor on the GPU
    x = torch.randn(1000000, device=device)

    # Set the lower and upper bounds for clamping
    lower = -0.5
    upper = 0.5

    # Warm-up run (to avoid including CUDA initialization time)
    _ = torch.clamp(x, lower, upper)

    # Ensure all previous CUDA calls have completed
    torch.cuda.synchronize()

    # Perform the clamp operation
    result = torch.clamp(x, lower, upper)

    # Ensure the operation is complete before ending the program
    torch.cuda.synchronize()

    print("Clamp operation completed")

if __name__ == "__main__":
    main()
