import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import numpy as np
import requests
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

#! Převod na správné zařízení
device = "cuda" if torch.cuda.is_available() else "cpu"

n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()



#! Train split
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#! Dělení
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,   #? 0,2 -> 20% jako testova data
                                                    random_state=42)
print(X_train[:5], y_train[:5])

#! Výpočet přesnosti
#? kolik budemít náš model správných výsldků ze celku
def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc=(correct / len(y_pred)) * 100
    return acc

#! Konstrukce modelu s nelinearní aktivační funkcí
from torch import nn
class CirculeModelV2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_1 = nn.Linear(in_features=2, out_features=10)  
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

        self.relu =nn.ReLU()    #? Nelinearní aktivační funkce
    
    def forward(self, x):
        #? umístění nelinearní kativační funkce
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CirculeModelV2().to(device)
print(model_3.state_dict())

#! Optimizer a loss funkce
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(),
                            lr=0.01)

#! Trenik nelinearního modelu
#? random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#? na správné device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


#? loop
epochs = 200000

for epochs in range(epochs):
    model_3.train()

    #? Forwardpass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #? Loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    #? zero grad
    optimizer.zero_grad()

    #? Loss backward
    loss.backward()

    #? Step optimizer
    optimizer.step()

    #! Testing
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy_fn(y_true=y_test, 
                               y_pred=test_pred)
    if epochs % 100 ==0:
        print(f"Epoch: {epochs} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%") 

#! Predikce
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
print(y_preds[:10], y_test[:10])

from helper_functions import plot_predictions, plot_decision_boundary
#!Grafická vizualizace
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1,2,1)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)
plt.show()


