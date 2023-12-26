import torch

# indexing

x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)
print(x[:, :, 2])   # : všechny části dané dimenze
print(x[:, 2, 2])   # : zanecha hranate zavorky
