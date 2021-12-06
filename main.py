import torch
from experiments import compare, gem

# CUDA for PyTorch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# compare.fit(device)
gem.fit(device)
