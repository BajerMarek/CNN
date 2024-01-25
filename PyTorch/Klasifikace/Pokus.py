import torch
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np
import requests
from pathlib import Path
from helper_functions import plot_predictions, plot_decision_boundary

#! Převod na správné zařízení
device = "cuda" if torch.cuda.is_available() else "cpu"

#! Stažení pomocných funkcí pokud nejsou již stažené
if Path("helper_functions.py").is_file():
    print("helper_functions.py již staženo, nestahuji znovu")
else:
    print("Stahuji helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

#! Vytvoření dat
weight = 0.7
bias =0.3
start = 0
end = 1
step = 0.01
X_regresion = torch.arange(start, end, step).unsqueeze(dim=1) # usqueeze -> vytvoří z každé hodnoty list 
y_regresion = weight * X_regresion + bias   #? zjednodušená linarní linearni rovnice

#! Check the data
print(len(X_regresion))
print(X_regresion[:5], y_regresion[:5])

#! Train split
train_split = int(0.8 * len(X_regresion))
X_train_regresion, y_train_regretion = X_regresion[:train_split],y_regresion[:train_split]
X_test_regresion, y_test_regresion = X_regresion[train_split:],y_regresion[train_split:]

#? kontrola dat
print (len(X_train_regresion),len(X_test_regresion),len(y_train_regretion),len(y_test_regresion))

#! Vizualizace
plot_predictions(train_data=X_train_regresion,
                 train_labels=y_train_regretion,
                 test_data=X_test_regresion,
                 test_labels=y_test_regresion)
plt.show()

#! Model za pomoci nn.Sequential (jinak stejný jako model_1)
model_2 = nn.Sequential(
    nn.Linear(in_features=1,out_features=10),
    nn.Linear(in_features=10,out_features=10),
    nn.Linear(in_features=10,out_features=1)
)

#! Loss a Optimizer
loss_fn = nn.L1Loss()
optimzer = torch.optim.SGD(params=model_2.parameters(),
                           lr=0.0001)

#! Trainig model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

X_train_regresion, y_train_regretion = X_train_regresion.to(device), y_train_regretion.to(device)
X_test_regresion, y_test_regresion = X_test_regresion.to(device), y_test_regresion.to(device)

#! Trainig
epoch = 20000

for epoch in range(epoch):
    y_pred = model_2(X_train_regresion)
    loss = loss_fn(y_pred, y_train_regretion)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    #? Testing
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regresion)
        test_loss = loss_fn(test_pred, y_test_regresion)

    #? Vizualizace (text)
    if epoch % 100==0:
        print(f"Epoch : {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")


model_2.eval()

#! Predikce
with torch.inference_mode():
    y_preds = model_2(X_test_regresion)

#! Grafická vizualizace
from helper_functions import plot_predictions, plot_decision_boundary
plot_predictions(train_data=X_train_regresion,
                 train_labels=y_train_regretion,
                 test_data=X_test_regresion,
                 test_labels=y_test_regresion,
                 predictions=y_preds)
plt.show()









