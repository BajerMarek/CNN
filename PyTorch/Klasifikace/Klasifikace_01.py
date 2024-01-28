# klasifikace pomocí neuronové sítě
from typing import Self
import torch
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np
import pathlib 
from pathlib import Path
import pandas as pd
#! Data
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import requests
#! oddělí text caramy
def print2(*args):
    print("=======================================")
    for arg in args:
        print(arg)
    print("=======================================")
#! vytvoření kol 1000 (kružnic)
n_samples = 1000
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
print(len(X), len(y))
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

#! Make DataFrae of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "labels": y})
print(circles.head(10))

#! Vizualizace - graficky
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.show()

#! 1.1 Převod dat do tensoru
print(f"Tvar X: {X.shape} | Tvar y: {y.shape}")

#! Nahled na příklad vzorku vlastností a jmen
X_sampel = X[0]
y_sampel = y[0]
print(f"Hodnota pro X : {X_sampel} to stejné pro y: {y_sampel}")
print(f"Tvar X_sampel: {X_sampel.shape} | Tvar y_sampel: {y_sampel.shape}")

#! 1.2 převod + rodělení dat na treniková a na testová
X = torch.from_numpy(X).type(torch.float) #? převede s numpy array do torch tensoru
y = torch.from_numpy(y).type(torch.float) #? to stejné pro y
print2(X[:5], y[:5])

#! Dělení dat
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,   #? 0,2 -> 20% jako testova data
                                                    random_state=42)
print2(len(X_train), len(X_test), len(y_train), len(y_test))

#! Převod na správné zařízení
device = "cuda" if torch.cuda.is_available() else "cpu"

#!optimalizovaný model
#? co se zmnění:
#? z 5 -> 10 neuronů
#? z 2 -> 3 vrstvy
#? z 100 -> 1000 epoch

#! Tvorba optimalizovaného modelu


class CirculeModelV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)  
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    def forward(self,x):
        #z = self.layer_1(x)
        #z = self.layer_2(z)
        #z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))  #? richlejší sintax
model_1 = CirculeModelV1().to(device)

#! Loss funkce
loss_fn = nn.BCEWithLogitsLoss()

#! Výpočet přesnosti
#? kolik budemít náš model správných výsldků ze celku
def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc=(correct / len(y_pred)) * 100
    return acc


#! Optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.0001)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

#! Trainig loop

#? Data na správné zařízení
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epoch = 50000
for epoch in range(epoch):
    model_1.train()
    #?Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    #? Loss
    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    #? Optimizer zero grad
    optimizer.zero_grad()
    #? Backpropagation
    loss.backward()
    #? Optimizer step
    optimizer.step()
    #! Testing
    model_1.eval()
    with torch.inference_mode():
        #?Forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        #? Loss
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
    if epoch % 100 ==0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc}%") 

#! Stažení pomocných funkcí pokud nejsou již stažené
if Path("helper_functions.py").is_file():
    print("helper_functions.py již staženo, nestahuji znovu")
else:
    print("Stahuji helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

#! Grafivká vizualizace

from helper_functions import plot_predictions, plot_decision_boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()

