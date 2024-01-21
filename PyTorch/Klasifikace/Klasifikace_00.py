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

#TODO stavba modelu - postup
#_ 1. Nastavenení spravného zařízení kde kód poběží
#_ 2. Konstrukce modelu (pomocí nn.Module)
#_ 3. Definování loss funkce a optimizeru
#! Tvorba modelu
device = "cuda" if torch.cuda.is_available() else "cpu"

#TODO konstrukce modelu
#_ 1. Subclass nn.Module (větši na modelu to dělá)
#_ 2. Vyrvoření 2 linearnich vrstev nn.Linear (schpných spracovat naše data)
#_ 3. Definování forward funkce
#_ 4. Nastaveni modelu na gpu

#! Samotná konstrukce
class CircleModelV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #? Vytvoření linearních vrstev
        self.layer_1 = nn.Linear(in_features=2, out_features=5)  #? definuje počet přijímaných dat a počet dat co vypočítá
        self.layer_2 = nn.Linear(in_features=5, out_features=1)  #? obě dvě vrstvy sou takzvané hiddne vrstvy
        #? definování forward funkce (která popisuje forward pass()
    def forward(self, x):
        return self.layer_2(self.layer_1(x))  #? x -> layer1 -> layer2
    
#! přesun modelu na správné zařízení (místo kde se bude počítat)
model_0 = CircleModelV0().to(device)
#print2(model_0)

#! Rychlejší forma konstrukce modelu (optimalizace procesu)
"""
model_1 = nn.Sequential(    #? model_1 == CircleModelV0
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
"""
#! Predikce s model_0
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

print2(f"Lenght of predictions: {len(untrained_preds.shape)}, Shape: {untrained_preds.shape}")
print2(f"Lenght of test samples: {len(X_test)}, Shape: {X_test.shape}")
print2(f"\nFirst 10 predictions:\n{torch.round(untrained_preds[:10])}")
print2(f"\nFirst 10 labels:\n{y_test[:10]}")

#!loss funkce a optimizer
#todo Kterou loss funkci vybrat?
#_ Vždy záleží na specifickém problému
#_ Pro linearni regresi -> MAE nebo MSE (mean absolute error / mean squared error)
#_ Pro klacifikaci -> binary cross entropy or categorical cross entropy (cross entropy) 
#? Pro připomenutí - loss funkce počítá nesprávnost výpočtu sítě

#todo Jaký optimizer vybrat?
#_ Zase záleží na daném problému problému 
#_ Nejpoužívanější (pro kategorizaci) jsou SGD(Stochastic gradient descent) a Adam
#_ Každopádně PyTorch má spoustu vbudovaných optimizeru takze stací vybrat na základě potřeby
#_

#! Nastavení loss funkce optimizeru
loss_fn = nn.BCEWithLogitsLoss()  #? nn.BCELoss() = potřebuje hodnoty které uz prosli přes sigmoid
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

#! Výpočet přesnosti
#? kolik budemít náš model správných výsldků ze celku
def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc=(correct / len(y_pred)) * 100
    return acc

#! Traing

#todo Postup
#_1. Forward pass
#_2. Výpočet loss
#_3. Optimizer zero grad
#_4. Loss backward (backpropagation)
#_5. Optimizer (gradiant descent)

#todo Přechod z čistích logitu -> předpovězené šance -> předpovězené popisy
#todo Going from raw lgits -> prediction prapabilities -> prediction labels 
#_ Výsledky našeho modelu budou čisté /raw logity
#_ Logity mužeme převést na prediction prapabilities (předpovězené šance) použitím 
#_  nějáké aktivační funkce (Sigmoid -> binarní klasifikace a softmax -> multiclass klasifikace)
#_ Následně můžeme prediction prapabilities(předpovězené šance) převést na prediction labels(předpovězené popisy)
#_  pomocí zaokrouhlení nebo argmax()

#! nahled na prvních 5 hodnot forward pass
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)

#! použití sigmoidu na logity
y_pred_probs = torch.sigmoid(y_logits)  #? přemnění logyty na prediction prapabilities
print(y_pred_probs)
print(torch.round(y_pred_probs))

#todo Pro naše hodnoty aby byly prediction prapabilitiess tak nanich musíme provézd zaokrouhleni na celá čísla
#? y_pred_probs >= 0.5 y=1 (class 1)
#? y_pred_probs < 0.5 y=0 (class 0)

#! Hledání prediction labels 
#? zaokrouhleni
y_preds = torch.round(y_pred_probs)
#? prediction labels        lgits -> prediction prapabilities -> prediction labels
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
#? zbaveni se zbytečne dimenze
print(y_preds.squeeze())

#! Trainig loop
torch.manual_seed(42)
#torch.cuda.manual_seed(42)
epochs = 500
#?  převod dat  na device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

#?samotný loop
for epochs in range(epochs):
    model_0.train()
    #? forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  #? zmněna z logitu na šance předpovědi na popisy
    #? výpočet loss a přesnosti
    loss = loss_fn(y_logits,    #? nn.BCEWithLogitsLoss potřebuje jako vstupní data logity
                   y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)  
    #? Optimizer zero grad
    optimizer.zero_grad()
    #? Loss backward   (backpropagation)
    loss.backward()
    #? Optimizer step   (gradient desent)
    optimizer.step()
    #! Testing
    with torch.inference_mode():
        #? Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        #? Výpočet test loss a accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
    #! Vizualizace
    if epochs % 10 ==0:
        print(f"Epoch: {epochs} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc}%") 
        
#todo předpovědi a hodnocení mmodelu
#_  Vypadá to že se  model nic neučí jen typuje a ještě špatně
#_  Řešení -> grafická vizualizace pro odhalení problému
#_  Aby jsem to dokazali importujeme funkci "plot_decision_boundary()"
#_  Vytáhneme si ji z githabu
        
#! Stažení pomocných funkcí pokud nejsou již stažené
if Path("helper_functions.py").is_file():
    print("helper_functions.py již staženo, nestahuji znovu")
else:
    print("Stahuji helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)
from helper_functions import plot_predictions, plot_decision_boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()

