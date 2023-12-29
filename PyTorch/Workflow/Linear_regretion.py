from typing import Self
import torch
from torch import nn # nn -> vše co je v PyTorch pro neuronové sítě
import matplotlib.pyplot as plt

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

# Trainig loop a testing loop
"""
0. Proskenování dat
1. Forward pass (prohnání dat skrz metodu forward()) - také nazívané jako forwad propagation
    za cílem vytvořit nějákou před předpovědí
2. Zpočítání loss funkce - (srovnání výsledku modelu s správnýmy výsledky)
3. Optimazer zero grad  nastavý hodnotu optimazeru na 0 aby se nezpomaloval výpočet (jak? nevím)
4. Loss backkwards -> pohybuje se od výsledku aby zjistíl spád jednolivých parametrů
   => (spád -> míra významnosti pro výsledek)  - (Back propagation)
5. Optimazer step - Využití optimizeru k upravě parametrů tak aby se snížila hodnota 
    z loss funkce. - (Gradiant desent)
"""
# epocha = 1 opakování ciklu
epochs = 1000
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
        print(f"Epoch: {epochs} | Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())                                      # 5. Optimizer step

with torch.inference_mode():
    y_preds = model_0(X_test)
plot_prediction(predictions=y_preds)
plt.show()
"""
Co model LinearRegressionModel dělá?
1. Ze začátku dostane náhodná čísla.
2. Podívá se na treniková data a pak upravý svá náhodná čísla tak aby se jeho výsledná data 
   co nejépe podobala trenigovým datům.

Jak se to děje?
1. Gradiant desent (algoritmus)
2. Backpropagation (algoritmus)
pomocí těchto algoritmů upravujeme ná gradiant
"""