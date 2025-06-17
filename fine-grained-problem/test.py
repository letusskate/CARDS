import torch
import matplotlib

# 检查PyTorch是否可以使用CUDA
if torch.cuda.is_available():
    print("CUDA is available. Device name:", torch.cuda.get_device_name())
else:
    print("CUDA is not available.")
