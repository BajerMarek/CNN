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

#Známé parametry
weight = 0.7
bias =0.3

# Vytvoření dat
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) # usqueeze -> vytvoří z každé hodnoty list 
y = weight * X + bias

print(X[:10], y[:10], len(X), len(y))

#Rozděleni dat na trénigové a testové data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]

X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

# Vizualizace
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
    plt.scatter(test_data, test_labels, s=4, label="Testing data")
    # mamé nějáké predikce?
    if predictions is not None:
        # pokud nějaké predikce exitují vykresli je červeně
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
plot_prediction()
#plt.show()

# vytvoření linearne regeresivního modulu
class LinearRegressionModel(nn.Module): # <- skoro vše (co se týče neuronových sítí) vpřebírá z nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,      # vytvoří parametr weights
                                                requires_grad=True, # získá gradinat
                                                dtype=float))
        self.bias = nn.Parameter(torch.randn(1,         # vytvoří parametr bias
                                             requires_grad=True,    # získá gradinat
                                             dtype=torch.float))
        # forward metoda určuje počítání modelu
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" je input
        return self.weights * x + self.bias     # Vzorec pro linearni regresi Y = X * B + e
# mytvoření seedu
torch.manual_seed(42)

# vyvoření modelu (subclass of nnmodel)
model_0 = LinearRegressionModel()

# první předdpovědí y_test na základě X_test
# když posíláme data přěs náš model tak forward data spracovává
# predikce
with torch.inference_mode():
    y_preds = model_0(X_test)
plot_prediction(predictions=y_preds)
#plt.show()
print(model_0.state_dict())
# nastavení loss funkce
loss_fn = nn.L1Loss()

# nastavení optimizeru      SGD -> stochastick gradient desent
optimizer = torch.optim.SGD(params=model_0.parameters(),    # co se má optimalizovat
                            lr=0.01)    # nejduležitější hyperparametr který si musíme sami naastavit

# epocha = 1 opakování ciklu
epochs = 200
                                                            # 0. Proskenovat dat
for epochs in range(epochs):
    model_0.train() # nastavý všechny parametry které potřebukí sklon aby potřebovaly sklon
    y_pred = model_0(X_train)                               # 1. Forward pass
    loss = loss_fn(y_pred,y_train)                          # 2. Loss funkce
    optimizer.zero_grad()                                   # 3. Optimazer zero grad
    loss.backward()                                       # 4. Back propagation
    optimizer.step()
    # testink loop
    # vypně všechny funkce které by zpomalovalí testování
    model_0.eval()
    with torch.inference_mode():    # vypne hlídání gradiantu   take se občas použiva torch.no_grad
        test_pred = model_0(X_test)                                 # 1. Forward pass
        test_loss = loss_fn(test_pred,y_test)                       # 2. Loss funkce   
    if epochs % 10 ==0:
        epoch_count.append(epochs)
        loss_value.append(loss)
        test_loss_value.append(test_loss)
        print(f"Epoch: {epochs} | Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())                                      # 5. Optimizer step

with torch.inference_mode():
    y_preds = model_0(X_test)
plot_prediction(predictions=y_preds)
plt.show()

plt.plot(epoch_count, np.array(torch.tensor(loss_value).cpu().numpy()), label="Train loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_value).cpu().numpy()), label=("Test loss"))
plt.title("trainig and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# ukládání modelu
# 1. kam chceme uložit
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Vytvoření cesty pro model
MODEL_NAME = "Model_workflow_01.pth"        # pth formát pro pythorch
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)
# 3. uložení parametrů
print(f"Ukladani modelu do: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# Pro nahrá ní našeho modelu musíme vytvořit nový model
loaded_model_0 = LinearRegressionModel()
print(loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH)))
print("------------------------------------------")
print(loaded_model_0.state_dict())
# vytvoření nějákých predikcí s nahraným modelem
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_0_preds = loaded_model_0(X_test)

print(loaded_model_0_preds)
# srovnání predikcý nahraného a původního modelu
print(y_preds == loaded_model_0_preds)
