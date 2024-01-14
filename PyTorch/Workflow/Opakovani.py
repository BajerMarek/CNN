from typing import Self
import torch
from torch import nn # nn -> vše co je v PyTorch pro neuronové sítě
import matplotlib.pyplot as plt
import numpy as np
import pathlib 
from pathlib import Path

# Data
weight = 0.5
bias = 0.09

X = torch.arange(0, 300, 1).unsqueeze(dim=1)
y = weight * X + bias
train_split = int(len(X) * 0.8)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(X_test))

# vizualizace
def plot_prediction(train_data=X_train,
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    predictions=None):   

#Vykresluje testová a trenigová data a porovnává predikce.

    plt.figure(figsize=(10, 7))
# vykteslí trenigová data modře
    plt.scatter(train_data, train_labels, c="b", s=4, label="Trainig data")
# vykteslí trenigová data modře
    plt.scatter(test_data, test_labels,c="g", s=4, label="Testing data")
    # mamé nějáké predikce?
    if predictions is not None:
        # pokud nějaké predikce exitují vykresli je červeně
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
plot_prediction()
plt.show()

# Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights =nn.Parameter(torch.randn(1,
                                   requires_grad=True,
                                   dtype=torch.float))
        self.biases = nn.Parameter(torch.randn(1,
                                  requires_grad=True,
                                  dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:             ###
        return self.weights * x + self.biases
model = LinearRegressionModel()


# Loss funkce
loss_fn = nn.L1Loss()                      ###

# Optimizer
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.001)        ###

# Predikce, trénik
torch.manual_seed(42)

epochs = 40000
for epochs in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()              ###
    loss.backward()                    ###
    optimizer.step()                   ###
    with torch.inference_mode():
        model.eval()
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred, y_test)
    if epochs % 4000 == 0:
        print(f"Epocha: {epochs} | Loss: {loss} | Test loss: {test_loss} | Parametry: {model.state_dict()}")

# Vitualizace_02
with torch.inference_mode():
    y_preds = model(X_test)
plot_prediction(predictions=y_preds)
plt.show()