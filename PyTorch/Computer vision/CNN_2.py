import torch
from torch import nn
#! Torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
#! Vizualizace
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import mlxtend
import torchmetrics
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
#! Další
import requests
from pathlib import Path
from timeit import default_timer as timer

import random
#! Device
device = "cuda" if torch.cuda.is_available() else "cpu"

#! Datasety
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

#! Dataloadery
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
class_names = train_data.classes

#! Čas
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """ Vypíše čas od začátku do konce"""
    total_time = end - start,
    print(f"Train time on {device}: {total_time} sekund")
    return total_time

#! Model
class FasionMNISTModelV4(nn.Module):
    """
    Konsturukce stejná jako TinyVGG
    ze stránky CNN Explainer
    """
    def __init__(self, input_shape : int, hidden_units : int, output_shape : int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),    #? Hyperparametry
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,       #! protože po Flatten ma tensor zpatne dimenze
                                                          #! 7 * 7 -> protoze po provedení maxpool tak se data z komprimují na dimeze 1,10,7,7
                      out_features=output_shape)
        )
    def forward(self,x):
        x = self.conv_block_1(x)
        #print(f"output shape of conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        #print(f"output shape of conv_block_2: {x.shape}")
        x = self.classifier(x)
        #print(f"output shape of classifier: {x.shape}")
        return x

#! inicializace
torch.manual_seed(42)                       
model_4 = FasionMNISTModelV4(input_shape=1,                                 #? pocet barev
                             hidden_units=10,                               #? Pocet neuronu
                             output_shape=len(class_names)).to(device)      #? Pocet mozných výsledku
print("bez chyb")

image, label = train_data[0]
plt.imshow(image.squeeze(), cmap="gray" )
plt.show()

#todo Pokus

rand_image_tensor = torch.randn(size=(1,28,28))
print(rand_image_tensor.shape)
print(model_4(rand_image_tensor.unsqueeze(0).to(device)))


#! Loss funkce eval metrics optimizer
from helper_functions import accuracy_fn
from Moje_funkce_CV import  train_step,test_step,eval_model,make_predictions

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)                         #! LR
#! funkce
#! Learnig loop
torch.manual_seed(42)
treain_time_start = timer()
epochs = 3
for epochs in tqdm(range(epochs)):
    print(f" Epoch: {epochs}\n================")
    train_step(model=model_4,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device
               )
    test_step(model=model_4,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
treain_time_end = timer()
total_train_time_model_4 = print_train_time(start=treain_time_start,
                                            end=treain_time_end,
                                            device=device)
model_4_results = eval_model(model=model_4,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)
print(model_4_results)

#! Vizualizace klasifikace modelu
random.seed(42)
test_samples =[]
test_labels = []
for sample, label in random.sample(list(test_data),k=9):    #? dělá 9 nahodných hodnot
    test_samples.append(sample)
    test_labels.append(label)
#print(test_samples[0].shape)
plt.imshow(test_samples[0].squeeze(),cmap="gray")
plt.title(class_names[test_labels[0]])
plt.show()
pred_probs = make_predictions(model=model_4,
                              data=test_samples)

#? zhlédnutí prvních par výsledků
#print(pred_probs[:2])

#? prediction propabilities to labels
pred_classes = pred_probs.argmax(dim=1)
#print(pred_classes)

#todo Grafická vyzualizace
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i,sample in enumerate(test_samples):
    #? create subplot
    plt.subplot(nrows,ncols,i+1)

    #? plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")
    
    #? predikce - nazev
    pred_label = class_names[pred_classes[i]]

    #? opravdový název
    truth_label = class_names[test_labels[i]]

    #? název vizualizace
    title_text = f"Předpověď: {pred_label} | Realita: {truth_label}"
    
    #? zmněna barvy textu pro predikované a pro realné názvy
    if pred_label == truth_label:
        plt.title(title_text, c="g")
    else:
        plt.title(title_text, c="r")
plt.axis(False)
plt.show()
#! Confuzion matrix -> pro lepšívyhodnocení
#todo Postup:
#_  1) Predikce vytvořené modelem na testovém datasetu
#_  2) Vytvoření Confusional matrix pomocí torchmatrix
#_  3) Zobrrazení Confusinonal matrixu pomocí mlxtend.plotting.plot_confusion_matrix()

#! 1)
y_preds = []
model_4.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Dělaní predikcí"):
        #? data na device
        X, y = X.to(device),y.to(device)
        #? forward pass
        y_logit = model_4(X)
        #? Predictions from logits -> predictions propabilities -> prediction lables
        y_pred = torch.softmax(y_logit.squeeze(),dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())
print(y_preds)
y_pred_tensor =torch.cat(y_preds)   #? převede list predikcí na jeden jediný tensor
print(y_preds[:10])

#? Setup confusin instance and compare predictions to target
confmat = ConfusionMatrix(task="multiclass",num_classes=len(class_names))
confmat_tensor = confmat(preds=y_pred_tensor,
                         target= test_data.targets)

#? plot confmat
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),            #? kvuli matplotibu
    class_names=class_names,
    figsize=(10,7)
)
plt.show()

#! Ukládání a nahrávání modelu
#? místo ulozeni
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#? Cesta pro model
MODEL_NAME = "CNN_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(MODEL_SAVE_PATH)

#? uložení parametrů
print(f"ukládání modelu do: {MODEL_SAVE_PATH}")
torch.save(obj=model_4.state_dict(),
           f=MODEL_SAVE_PATH)
print("hotovo")

#! Test nahraného modelu
torch.manual_seed(42)

loaded_model = FasionMNISTModelV4(input_shape=1,
                                  hidden_units=10,
                                  output_shape=len(class_names))

#? nahrání modelu
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model.to(device)

#! Vyhodnocení nahrného modelu
print(model_4_results)
torch.manual_seed(42)
loaded_model_results = eval_model(model=loaded_model,
                                  data_loader=test_dataloader,
                                  loss_fn=loss_fn,
                                  accuracy_fn=accuracy_fn)
print(loaded_model_results)
#? zjištějí pomocí pytorch jestly jsou výsledky stejné
print(torch.isclose(torch.tensor(model_4_results["model_loss"]),
              torch.tensor(loaded_model_results["model_loss"]),
              atol=1e-02) )  #? míra tolerance
