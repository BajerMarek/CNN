import torch
print("ahoj")
print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
print(torch.cuda.device_count())

tensor = torch.tensor([1, 2, 3])    # aoutomaticky na CPU

tensor_on_gpu = tensor.to(device)   # přesune tensor na GPU
print(tensor_on_gpu)                # Numpy funguje pouze na CPU
