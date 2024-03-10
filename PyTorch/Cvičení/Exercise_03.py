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
import numpy as np

#! device
device = "cuda"if torch.cuda.is_available() else "cpu"

#! data
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)
test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None,
    download=True
)
#! Vizualizace
fig, axs = plt.subplots(1,5, figsize=(12, 5))
for i in range(5):
    x = random.randint(0,16000)
    image,label = train_dataset[x]

    axs[i].imshow(image.squeeze(), cmap="gray")
plt.tight_layout()
plt.show()
#! Dataloadery
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              )
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
class_names = train_dataset.classes
#! Model
class FasionMNISTModelV5(nn.Module):
    """
    Konsturukce stejná jako TinyVGG
    ze stránky CNN Explainer
    """
    def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
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
            nn.MaxPool2d(kernel_size=2),

            )

        self.block_2 = nn.Sequential(
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
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
            )
    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x
#! Inicializace modelu                                   
torch.manual_seed(42)
model_5 = FasionMNISTModelV5(input_shape=1,
                             hidden_units=10,
                             output_shape=len(class_names)).to(device)

#! Loss + optimizer + Moje funkce
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_5.parameters(),
                            lr=0.1)
from helper_functions import accuracy_fn
from Moje_funkce_E import train_step, test_step, print_train_time,eval_model,make_predictions

star_timer = timer()
epochs = 3
model_5.train()
for epochs in tqdm(range(epochs)):
    print(f" Epoch: {epochs}\n================")
    train_step(model=model_5,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_5,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
end_timer = timer()
print_train_time(start=star_timer,
                 end=end_timer,
                 device=device)
model_5.eval()
eval_model(model=model_5,
           data_loader=test_dataloader,
           loss_fn=loss_fn,
           accuracy_fn=accuracy_fn)

#! Vizualizace
test_samples = []
test_lables = []
for sample, label in random.sample(list(test_dataset),k=15):
    test_samples.append(sample)
    test_lables.append(label)
plt.imshow(test_samples[0].squeeze(),cmap="gray")
plt.title(class_names[test_lables[0]])
plt.show()
pred_probs = make_predictions(model=model_5,
                              data=test_samples)

#? prediction propabilities to labels
pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(12,12))
nrows = 3
ncolums = 5
for i, sample in enumerate(test_samples):
    #? create subplot
    plt.subplot(nrows,ncolums, i+1)
    #? plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")
    #? predikce
    pred_label = class_names[pred_classes[i]]
    #? realita
    truth_label = class_names[test_lables[i]]
    #? název
    title_text = f"Předpověď: {pred_label} | Realita: {truth_label}"
    #? zmněna barvy
    if pred_label == truth_label:
        plt.title(title_text, c="g")
    else:
        plt.title(title_text,c="r")
plt.axis(False)
plt.show()
#! Confuzion matrix
y_preds = []
model_5.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader,desc="Dělení predikcí"):
        #? device
        X,y = X.to(device),y.to(device)
        #? forward pass
        y_logit = model_5(X)
        #? predictions from logits ->predictions propabilities -> prediction labels
        y_pred = torch.softmax(y_logit.squeeze(),dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())    #? pro matplotlib    
y_pred_tensor = torch.cat(y_preds)
#? porovnávání perdikcí s realitou
confmat = ConfusionMatrix(task="multiclass",num_classes=len(class_names))
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_dataset.targets)

#? zobrazení
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),     #? pro matplotlib
    class_names=class_names,
    figsize=(10,7)
)
plt.show()
