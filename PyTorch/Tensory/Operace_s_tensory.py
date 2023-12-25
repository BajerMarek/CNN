import torch


# operace s tensory

tensor = torch.tensor([1, 2, 3])
print(tensor + 10)   # přidá ke každé hodnotě v tensoru 10
print(tensor * 10)   # výnásobí každou hodnotu v tensoru 10
print(tensor - 10)   # odečte 10 od každé hodnoty v tensoru

# násobení matrixů

# element-wise
print(tensor, "*", tensor)
print(f"Se rovná:{tensor * tensor}")    # vynásobí první hodnotu s první a druhou s drouhou...

# matrix mutiplication
print(torch.matmul(tensor, tensor))     # vynásobí první hodnotu s první a přičte  druhou s drouhou...

# Tvary pro nasobení matrixu
tensor_A = torch.tensor([[1, 2],
                        [3, 4],
                        [5, 6]])

tensor_B = torch.tensor([[7, 10],
                        [8, 11],
                        [9,12]])
print(tensor_B)
print(tensor_B.T)   # přehodí sloupece((7, 8, 9) a (10, 11, 12) ) v tensor_B na řádek
print(torch.matmul(tensor_A, tensor_B.T))
# vizualizace
print(f"Original shapes: tensor_A = {tensor_A}, tenosr_B = {tensor_B}")
print(f"New shapes: tensor_A = {tensor_A} (same shape as above), tensor_B.T = {tensor_B.T}")
print(f"multiplying:{tensor_A.shape} @ {tensor_B.T.shape} <- inner dimesion must mach")
print("output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape: {output.shape}")
