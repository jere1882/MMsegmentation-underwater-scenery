import torch

def validate_gpu_setup():
    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your CUDA installation.")
        return

    # Get CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")

    # List available CUDA devices
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_gpus}")

    for i in range(num_gpus):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"  Memory cached: {torch.cuda.memory_reserved(i)} bytes")
    
    # Perform a simple tensor operation on GPU
    try:
        device = torch.device("cuda:0")
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        c = a + b
        print(f"\nSimple tensor operation result: {c}")

        if torch.all(c == torch.tensor([5.0, 7.0, 9.0], device=device)):
            print("GPU is functioning correctly for tensor operations.")
        else:
            print("Unexpected result from tensor operation.")
    except Exception as e:
        print(f"An error occurred during the tensor operation: {e}")

# Run the validation
validate_gpu_setup()

