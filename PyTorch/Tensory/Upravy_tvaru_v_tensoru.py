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
print("-----------------------------------------------------------")

# Odstranění dimenze z tensoru
print(x_reshaped.shape)
print(x_reshaped.squeeze().shape)
print("-----------------------------------------------------------")
x_squeezed = x_reshaped.squeeze()
print(f"Před zmněnou: {x_reshaped}")
print(f"Tvar před zmněnou: {x_reshaped.shape}")
print("-----------------------------------------------------------")
print("-----------------------------------------------------------")
print(f"Po zmněně: {x_squeezed}")
print(f"Tvar po zmněně: {x_squeezed.shape}")
print("-----------------------------------------------------------")
x_unsqeeyed = x_squeezed.unsqueeze(dim=0)
print(f"Unsqueezed tenzor: {x_unsqeeyed}")
print(f"Tvar usqueezed tensoru: {x_unsqeeyed.shape}")
print("-----------------------------------------------------------")
print("-----------------------------------------------------------")
print("-----------------------------------------------------------")

# premute
x_original = torch.randn(size=(224, 224, 3))    # výška, šířka , barvy
# zmněnění pořadí
x_permuted = x_original.permute(2, 0, 1)        # barvy šířka výška

print(f"Před premute: {x_original.shape}")      # výška, šířka , barvy
print(f"Po premute: {x_permuted.shape}")        # barvy šířka výška


"""
y = torch.randn(1, 9, 3, 1)
print(y)
print(y.shape)
print("-----------------------------------------------------------")

print(y.squeeze())
print(y.squeeze().shape)
"""