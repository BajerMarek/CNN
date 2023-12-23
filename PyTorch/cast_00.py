import torch
import pandas as pd
import numpy as np
import matplotlib as plt
"""
### typy listů / nezorů
scalar = torch.tensor(7)
#print(scalar)

vector = torch.tensor([7, 7])
#print(vector)

matrix = torch.tensor([ [1, 2],
                        [3, 4]])
#print(matrix)

tensor = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])

random_tensor = torch.rand(3, 4)     # vytvoří tensor s náhodnými hodnotami
#print(random_tensor)
random_image_size_tensor = torch.rand(size=(224, 224, 3))  # výška šířka počet barev 
#print(random_image_size_tensor)
zeros = torch.zeros(3, 4)   # vytcoří tenzor plný 0
#print(zeros)
ones = torch.ones(3, 4)     # vytcoří tenzor plný 1
#print(ones)
#print(torch.arange(start=0, end=1000, step=77))     #začátek, konec, každý posun o kolik
one_to_ten = torch.arange(start=1, end=11, step=1)      # vytvoří řadu 0d 1 - 10
ten_zeros = torch.zeros_like(input=one_to_ten)      # přemnění řadu na nuli
#print(ten_zeros)

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,      # data typ tensoru, defaultně float32
                               device="cpu",    # kde je tensor uložen
                               requires_grad=False)     # jestli sledovat růst toot tensoru
#print(float_32_tensor.dtype)

float_16_tensor =  float_32_tensor.type(torch.float16)
#print(float_16_tensor)

int_32_tensor = torch.tensor([3, 5, 9],dtype=torch.int32)
#print(float_16_tensor*int_32_tensor)
"""
# získávání dat z tensoru
some_tensor = torch.rand(3, 4)
print(some_tensor)
print(f"Datatype of tensor:{some_tensor.dtype}") 
print(f"Shape of tensor:{some_tensor.shape}") 
print(f"Device of tensor:{some_tensor.device}") 

