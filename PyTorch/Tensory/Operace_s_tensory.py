import torch
import pandas as pd
import numpy as np
import matplotlib as plt

# operace s tensory

tensor = torch.tensor([1, 2, 3])
print(tensor + 10)   # přidá ke každé hodnotě v tensoru 10
print(tensor * 10)   # výnásobí každou hodnotu v tensoru 10
print(tensor - 10)   # odečte 10 od každé hodnoty v tensoru

# násobení matrixů

# element-wise
print(tensor, "*", tensor)
print(f"Se rovná:{tensor * tensor}")

# matrix mutiplication
print(torch.matmul(tensor, tensor))