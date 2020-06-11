import torch


cuda_available = torch.cuda.is_available()
torch.cuda.init()
cuda_init = torch.cuda.is_initialized()
cuda_device = torch.cuda.get_device_name()
print(f'cuda_available = {cuda_available}')
print(f'cuda_init = {cuda_init}')
print(f'cuda_device = {cuda_device}')