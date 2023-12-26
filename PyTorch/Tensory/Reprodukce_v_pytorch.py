import torch
import numpy as np
# tensory s nahodnými čsly
random_tensor_A = torch.randn(3, 4)
random_tensor_B = torch.randn(3, 4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

# náhodné tenzory které ale lze zreplikovat
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.randn(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.randn(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)


