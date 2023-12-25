import torch

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,      # data typ tensoru, defaultně float32
                               device="cpu",    # kde je tensor uložen
                               requires_grad=False)     # jestli sledovat růst toot tensoru
#print(float_32_tensor.dtype)

float_16_tensor =  float_32_tensor.type(torch.float16)
#print(float_16_tensor)

int_32_tensor = torch.tensor([3, 5, 9],dtype=torch.int32)
#print(float_16_tensor*int_32_tensor)

# získávání dat z tensoru
some_tensor = torch.rand(3, 4)
print(some_tensor)
print(f"Datatype of tensor:{some_tensor.dtype}") 
print(f"Shape of tensor:{some_tensor.shape}") 
print(f"Device of tensor:{some_tensor.device}") 
