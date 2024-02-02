from typing import Self
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn 
import pathlib 
from pathlib import Path
import requests
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
X_moon, y_moon = make_moons(n_samples=1000,
                            random_state=42)
X_moon = torch.from_numpy(X_moon).type(torch.float)                                         #_
y_moon = torch.from_numpy(y_moon).type(torch.float)                                         #_ 
    
X_moon_train,X_moon_test,y_moon_train,y_moon_test = train_test_split(X_moon,
                                                                     y_moon,
                                                                     test_size=0.2,
                                                                     random_state=42)
plt.figure(figsize=(10,7))                                                                  #_
plt.scatter(X_moon[:,0], X_moon[:,1], c=y_moon, cmap=plt.cm.RdYlBu)                         #_
plt.show()                                                                                  #_

class Model(nn.Module):
    def __init__(self,input_features, output_features, hidden_units=20):                    #_
        super().__init__()
    
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_units),
            nn.ReLU(),                                                                      #_
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_features),
    )
    def forward (self, x):                                                                  #_
        return self.layer_stack(x)

model = Model(input_features=2,
              output_features=1,)
torch.manual_seed(42)                                                                       #_

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.01)

def Accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc=(correct / len(y_pred)) * 100
    return acc

epochs = 10000
for epochs in range(epochs):
    model.train()
    y_logits = model(X_moon_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))                               #_
    loss = loss_fn(y_logits, y_moon_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = Accuracy(y_true=y_moon_train.type(torch.LongTensor),
                   y_pred=y_pred)
    model.eval()
    with torch.inference_mode():
        model.train()
        test_logits = model(X_moon_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_moon_test)
        test_acc = Accuracy(y_true=y_moon_test,
                            y_pred=test_pred)
        optimizer.zero_grad()
        optimizer.step()
    if epochs % 10 ==0:
        print(f"Epochs: {epochs} | Loss: {loss} | Přesnost: {acc}% | Test loss: {test_loss} | Přesnost testu: {test_acc}% |")

#! Stažení pomocných funkcí pokud nejsou již stažené
if Path("helper_functions.py").is_file():
    print("helper_functions.py již staženo, nestahuji znovu")
else:
    print("Stahuji helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Trenigova data")
plot_decision_boundary(model,X_moon_train,y_moon_train)
plt.subplot(1,2,2)
plt.title("Výsledek")
plot_decision_boundary(model,X_moon_test,y_moon_test)
plt.show()