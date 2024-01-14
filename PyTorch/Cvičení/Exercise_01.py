from typing import Self
import torch
from torch import nn # nn -> vše co je v PyTorch pro neuronové sítě
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from pathlib import Path

epoch_count = []
loss_value = []
test_loss_value = []

# parametry
weight = 0.3
bias = 0.9

# data set
X = torch.arange(0, 200, 1).unsqueeze(dim=1)
y = weight * X + bias

print (len(X), len(y))

traing_split = int(len(X) * 0.8)
y_trainig = y[:traing_split]
X_trainig = X[:traing_split]
y_test = y[traing_split:]
X_test = X[traing_split:]
print (len(X_trainig), len(y_trainig))
print (len(X_test), len(y_test))

#Vizualizace
def plot_prediction(train_data=X_trainig,
                    train_labels=y_trainig,
                    test_data=X_test,
                    test_labels=y_test,
                    predictions=None):

    plt.figure(figsize=(17, 7))
# vykteslí trenigová data modře
    plt.scatter(train_data, train_labels, c="b", s=4, label="Trainig data")
# vykteslí trenigová data modře
    plt.scatter(test_data, test_labels, c="g" , s=4, label="Testing data")
    # mamé nějáké predikce?
    if predictions is not None:
        # pokud nějaké predikce exitují vykresli je červeně
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
plot_prediction()
plt.show()

# model

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=float))
        self.bias = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
# vytvoří seed
torch.manual_seed(42)

model = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model(X_test)
plot_prediction(predictions=y_preds)

# loss funkce
loss_fn = nn.L1Loss()

# optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# trainig 
epochs = 300

for epochs in range(epochs):
    model.train()
    y_pred = model(X_trainig)
    loss = loss_fn(y_pred, y_trainig)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.inference_mode():
        model.eval()
        y_pred = model(X_test)
        test_loss = loss_fn(y_pred, y_test)
    if epochs % 20 == 0:
        epoch_count.append(epochs)
        loss_value.append(loss)
        test_loss_value.append(test_loss)
        print(f"Epoch: {epochs} | Loss: {loss} | Test loss: {test_loss}")

with torch.inference_mode():
    y_preds = model(X_test)
plot_prediction(predictions=y_preds)
plt.show()

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Exercise_01_model"

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

Loaded_model = LinearRegressionModel()
Loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print(Loaded_model.state_dict)
Loaded_model.eval()
with torch.inference_mode():
    Loadede_model_preds = Loaded_model(X_test)
    
print(y_preds == Loadede_model_preds)

