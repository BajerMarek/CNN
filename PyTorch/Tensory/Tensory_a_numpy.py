import torch
import numpy as np

# array -> tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)    # vyhodí hodnotuve formatu float64 -> long
print(array, tensor)
print(array.dtype)
print(tensor.dtype)

# tensor -> array
tensor = torch.ones(7)

numpy_tensor = tensor.numpy()
print(tensor + 1, numpy_tensor)