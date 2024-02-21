# můj pokus o nelinearní CV model
import torch
from torch import nn 
#! Torchvision
import torchvision 
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
#! Matplotlib
import matplotlib.pyplot as plt
#! Další
import requests
from pathlib import Path
from timeit import default_timer as timer
from tqdm.auto import tqdm 
#! Data set
#? Fasion mnist z torch vision
#? trainig data
train_data = datasets.FashionMNIST(
    root="data",                                      #? kam se stáhnou
    train=True,                                       #? Tranigový dataset ano / ne
    download=True,                                    #? Stáhnout ano / ne
    transform=torchvision.transforms.ToTensor(),      #? Transformaovat ano / ne / jak
    target_transform=None                             #? Jak transformovat labels / target
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
class_names = train_data.classes
#! Dataloadery
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              )
test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              )
#! Model
class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 hidden_units:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),                                                       #!
            nn.Linear(in_features=in_features,out_features=hidden_units),            
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),         
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),         
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=out_features)            
        )
    def forward(self,x: torch.Tensor):
        return self.layer_stack(x)
#! Inicializace
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(in_features=784,
                              hidden_units=32,
                              out_features=len(class_names))
from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_1.parameters(),
                            lr = 0.01)
epochs = 3

for epochs in tqdm(range(epochs)):
    print(f" Epoch: {epochs}\n======")
    train_loss = 0
    train_acc = 0
    for batch, (X,y) in enumerate(train_dataloader):                              #!
        model_1.train()
        train_pred = model_1(X)
        loss = loss_fn(train_pred,y)
        train_loss += loss
        acc = accuracy_fn(y_pred=train_pred.argmax(dim=1),
                          y_true=y)
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 500 == 0:
            print(f"Počet zpracovaných obrázků: {batch*len(X)}/{len(train_dataloader.dataset)}")
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    test_loss = 0
    test_acc = 0
    model_1.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dataloader:                                       #!
            test_pred = model_1(X_test)
            loss = loss_fn(test_pred, y_test)
            test_loss += loss
            acc = accuracy_fn(y_pred=test_pred.argmax(dim=1),
                              y_true=y_test)
            test_acc += acc
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
print(f"Train loss: {train_loss} | Train acc: {train_acc}% | Test loss: {test_loss} | Test acc: {test_acc}%")




