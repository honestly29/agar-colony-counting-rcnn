import torch

print("=== GPU STATUS CHECK ===")

# 1. CUDA available?
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Current device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

    # 2. Allocate some tensors on GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = torch.matmul(x, y)  # simple GPU compute

    # 3. Report GPU memory usage
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"GPU memory allocated: {allocated:.2f} MB")
    print(f"GPU memory reserved:  {reserved:.2f} MB")

else:
    print("No GPU detected by PyTorch")

