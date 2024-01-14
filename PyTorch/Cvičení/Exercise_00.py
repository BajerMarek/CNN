# zdroj uloh https://www.learnpytorch.io/00_pytorch_fundamentals/
import torch
import numpy as np
#                           1.
tensor_1 = torch.randn(7, 7)
print(tensor_1.shape)
#                           2.
tensor_2 = torch.randn(1, 7)

print(torch.matmul(tensor_1, tensor_2.T))
#                           3.
torch.manual_seed(0)
tensor_1 = torch.randn(7, 7)
torch.manual_seed(0)
tensor_2 = torch.randn(1, 7)

print(torch.matmul(tensor_1, tensor_2.T))
#                           10.
torch.manual_seed(7)
tensor_3 = torch.randn(1, 1, 1, 10)
tensor_3_squeezed = tensor_3.squeeze()
print(tensor_3_squeezed)
print(tensor_3_squeezed.shape)
print(tensor_3.shape)

