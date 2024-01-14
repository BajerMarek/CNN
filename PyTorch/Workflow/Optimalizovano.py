from typing import Self
import torch
from torch import nn # nn -> vše co je v PyTorch pro neuronové sítě
import matplotlib.pyplot as plt
import numpy as np
import pathlib 
from pathlib import Path

# Data
weight = 0.7
bias = 0.3

X = torch.arange(0, 100, 0.02).unsqueeze(dim=1)
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

class LinearRegressionModeV2(nn.Module):
    def __init__(self):
        super().__init__()
        # využití nn.Linear() pro vytvoření parametru
        self.linear_layer = nn.Linear(in_features=1,    # kolik hodnotot dostane vyrstva -> 1 X_train
                                      out_features=1)   # kolik hodnotot vyplinve vyrstva -> 1 y_train
    def forward(self, x: torch.Tensor) -> torch.Tensor:    # říkáže x má format torch tensor a ze funkce vrátí torch tensor
        return self.linear_layer(x)

torch.manual_seed(42)
model_2 = LinearRegressionModeV2()

"""
print(model_1, model_1.state_dict())
# určení device pro kod
print(next(model_1.parameters()).device)
model_1.to(device="gpu")
print(model_1, model_1.state_dict())
"""
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_2.parameters(), 
                            lr =0.01)

epochs = 20000

for epochs in range(epochs):
    model_2.train()
    # forward
    y_pred = model_2(X_train)

    # loss
    loss = loss_fn(y_pred, y_train)

    # optimizer
    optimizer.zero_grad()

    # back propagatino
    loss.backward()

    # optimizer step
    optimizer.step()

    # testing
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test)

        test_loss = loss_fn(test_pred, y_test)
    
    if epochs % 10 ==0:
        print(f"Epoch: {epochs} | Loss: {loss} | Test loss: {test_loss}")

# vizualizaace
with torch.inference_mode():
    y_preds = model_2(X_test)
plot_prediction(predictions=y_preds)
plt.show()
print(model_2.state_dict())
# umistení
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# cesta pro umisteni
MODEL_NAME = "Model_optim_workflow_02.pht"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# uloženi state_dict()
print(f"Saving model to : {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)

print(MODEL_SAVE_PATH)

# nahrání modelu
loaded_model_2 = LinearRegressionModeV2()

# nahrat state dict modelu
loaded_model_2.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Nahrani modelu na gpu
"""
loaded_model_2.to(device="gpu")
print(next(loaded_model_2.parameters()).device)
"""
# hodnocení modelu
loaded_model_2.eval()
with torch.inference_mode():
    loaded_model_2_preds = loaded_model_2(X_test)
print(y_preds == loaded_model_2_preds)