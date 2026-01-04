try:
    import torch
except ImportError:
    print("PyTorch is not installed. Please install it using 'pip install torch'.")
    exit(1)

device = "CPU"

if torch.version.cuda is not None and torch.cuda.is_available():
    device = "CUDA (NVIDIA GPU)"
elif hasattr(torch.version,
             "hip") and torch.version.hip is not None and torch.version.hip != "" and torch.version.hip != "None":
    try:
        if torch.cuda.is_available():
            device = "ROCm (AMD GPU)"
    except Exception:
        pass
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "MPS (Apple GPU)"
elif hasattr(torch.backends, "xpu") and torch.backends.xpu.is_available():
    device = "XPU (Intel GPU)"
else:
    device = "CPU"

print(f"The current device is {device}")
