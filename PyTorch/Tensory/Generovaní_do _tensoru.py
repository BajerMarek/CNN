import torch

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