
#! Replikace nelinearních funkcí dosud použitých v modelech
import torch
import matplotlib.pyplot as plt


A = torch.arange(-10, 10, 1, dtype=torch.float32)
plt.plot(A)
plt.show()

plt.plot(torch.relu(A))
plt.show()

#! ReLu
def relu(x):
    return torch.maximum(torch.tensor(0),x) #! input musí byt tensor
plt.plot(relu(A))
plt.show()

#! Sigmoid
def sigmoid(x):
    return 1/(1+torch.exp(-x))
plt.plot(torch.sigmoid(A))
plt.show()

plt.plot(sigmoid(A))
plt.show()







