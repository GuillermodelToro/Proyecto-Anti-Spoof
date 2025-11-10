import torch
print("is_available:", torch.cuda.is_available())
print("torch.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
