import torch
x = torch.arange(1., 10.)
# .reshape(První hodnota = počet řádků, Druhá hodnota = počet sloupců)
x_reshaped = x.reshape(1, 9)
print(x_reshaped, x_reshaped.shape)

# View
z = x.view(1, 9)
print(z, z.shape)
z[:,0] = 5
print(z, x)

# Stack tensory na sebe
x_stack = torch.stack([x, x, x, x], dim=0)
print(x_stack)