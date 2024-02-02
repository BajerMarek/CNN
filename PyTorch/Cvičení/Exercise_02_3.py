from typing import Self
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn 
from sklearn.model_selection import train_test_split

# Code for creating a spiral dataset from CS231n
N = 1000 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

X = torch.from_numpy(X).type(torch.Tensor)
y = torch.from_numpy(y).type(torch.Tensor)

X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                test_size=0.2,
                                                random_state=42)


  
class ModelV2 (nn.Module):
    def __init__(self,input_features,output_features,hidden_units=24) -> None:
        super().__init__()
        self.layer_stack= nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_features),
        )
    def forward(self, x):
       return self.layer_stack(x)
model_2 = ModelV2(input_features=2,
                  output_features=K)
torch.manual_seed(42)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_2.parameters(),
                             lr=0.0001)
def Accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc=(correct / len(y_pred)) * 100
    return acc

epochs = 20000
for epochs in range(epochs):
    model_2.train()
    y_logits = model_2(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_fn(y_logits, y_train.type(torch.LongTensor))
    acc = Accuracy(y_true=y_train,
                  y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(X_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test.type(torch.LongTensor))
        test_acc = Accuracy(y_true=y_test,
                            y_pred=test_pred)
        optimizer.zero_grad()
        optimizer.step()
    if epochs % 10 ==0:
        print(f"Epocha: {epochs} | Loss: {loss} | Přesnost: {acc} | Test loss: {test_loss} | Přesnost testu: {test_acc}")

from helper_functions import plot_predictions, plot_decision_boundary
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Trenik")
plot_decision_boundary(model_2,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_2,X_test,y_test)
plt.show()


