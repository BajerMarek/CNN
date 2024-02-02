from typing import Self
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn 
#! Data

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

#! Nastavené hyperparametrů
NUM_CLASSES = 4                #! zde upravyt v případě potřby pro více dat a níže v kódu též
NUM_FEATURES = 2
RANDOM_SEED =42

#! Vytvoření dat
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,    #? trochu "zatřese " zhluky
                            random_state=RANDOM_SEED)

#! Převod na tensory
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.float)

#! Rozdělení dat 
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

#! Vizualizace
plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()

#! Device
device = "cuda" if torch.cuda.is_available()else "cpu"

#! Tvorba multi-class klasifikačního modelu
class BlobModel(nn.Module):
    def __init__(self,input_features, output_features, hidden_units=8):
        """

        Args: 
        input features (int): Number of input features to the model
        output features (int): Number of outputs features (number of output classes)
        hidden units (int) : Number of hidden units between layers, default 8




        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)

        )   
    def forward (self, x):
        return self.linear_layer_stack(x)

model_4 = BlobModel(input_features=2,
                    output_features=4,                                  #! zde upravyt v případě potřby pro více dat
                    hidden_units=8).to(device)

#! Loss funkce a optimizer
#? Loss
loss_fn = nn.CrossEntropyLoss()

#? Optimizer
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)                                     #! Lr

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc=(correct / len(y_pred)) * 100
    return acc

model_4.eval()
with torch.inference_mode():
    y_logits =(model_4(X_blob_train.to(device)))
print(y_logits[:10])

#todo Pro fungování modelu musíme převést jeho output (logity) na pravděpodobnost a pak
#todo na názvy pro pravděpodobnost 
#todo Going from raw logits -> prediction prapabilities (soft-max) -> prediction labels (argmax)

#! převod na prediction prapabilities   (soft-max)
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])
print(y_pred_probs[:5])

#! Vlastnosti soft max
print(torch.sum(y_pred_probs[0]))   #? dohromady 1
print(torch.max(y_pred_probs[0]))   #? volba našeho modelu jako správné řešení
print(torch.argmax(y_pred_probs[0]))    #? index hodnoty která je jeho volbou

#! převod na prediction labels 
y_preds = torch.argmax(y_pred_probs,dim=1)
print(y_preds)
print(y_blob_test)

#! Trainig a testing loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#? počet opakování
epochs = 1000

#? data na spravne zařízení
X_blob_train,y_blob_train = X_blob_train.to(device),y_blob_train.to(device)
X_blob_test,y_blob_test = X_blob_test.to(device),y_blob_test.to(device)


#? loop
for epochs in range(epochs):
    model_4.eval()

    y_logits=model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train.type(torch.LongTensor))
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #? testing
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test.type(torch.LongTensor))
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_pred)
    #? vizualizace
    if epochs % 10 == 0:
        print(f"Epochs: {epochs} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")  

#! Predikce
#? logity / predikce
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
print(y_logits[:10])
#? logit -> prediction propability

y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_pred_probs[:10])
#? prediction propability -> pred labels
y_pred_probs = torch.argmax(y_pred_probs,dim=1)
print(y_pred_probs[:10])

#! vizualizace
from helper_functions import plot_predictions, plot_decision_boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()


#todo Možnosti vyhodnocení výkonu modelu
#_ Accuracy - ze 100 bodů kolik jich je správně
#_ Precision -
#_ Recall
#_ F1-score
#_ Confusion matrix
#_ Classification report
