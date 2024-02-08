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
#! Device
device = "cuda" if torch.cuda.is_available() else "cpu"
#! DAta sety
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
#! Dataloadery
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
class_names = train_data.classes
#! Model
class FashionMNISTModelV2(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                       out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                       out_features=output_shape),
            nn.ReLU()

        )
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=784,  
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
#! Loss funkce + optimizer
from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)
#! Čas
def print_train_time(start: float,
                     end: float,
                     device: torch.device =None):
    """ Vypíše rozdíl mezi časem začátku a koncem"""
    total_time = end -start
    print(f"Train time on {device}: {total_time} sekund")
    return total_time
#! Trainig loop (ve formně funkce)
def train_step(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               accuracy_fn,
               device:torch.device=device):
    """Vytvoří trainig loop pro zadaný model"""
    train_loss, train_acc = 0,0
    model.train()
    #! Trainig loop
    for batch, (X, y) in enumerate(data_loader):
        #? Převod dat na správné zařízení
        X,y = X.to(device),y.to(device)
        #? Forward pass
        y_pred = model(X)
        #? Loss per bach
        loss = loss_fn(y_pred,y)
        train_loss += loss      #? přidává do promněné 
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))
        #? Optimize zero grad
        optimizer.zero_grad()
        #? Back propagation
        loss.backward()
        #? Optimizer step
        optimizer.step()
        if batch % 400 == 0:
            print(f"Looked at {batch*len(X)}/{len(data_loader.dataset)} sampels.")

    #? Dělení train_loss  dékou dataloaderu(batches)
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    
#! Test loop (ve formně funkce)
def test_step(model:torch.nn.Module,
              data_loader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              accuracy_fn,
              device:torch.device = device):
    test_loss,test_acc = 0,0
    model.eval()
    for X_test,y_test in test_dataloader:
        #! Převod dat na správné zařízení
        X_test,y_test = X_test.to(device),y_test.to(device)
        #? Forward pass
        test_pred = model_2(X_test)                              #!
        #? Test loss
        test_loss += loss_fn(test_pred,y_test)                    #!
        #? Přesnost
        test_acc += accuracy_fn(y_true=y_test,y_pred=test_pred.argmax(dim=1))   #!
        #? Průměená test loss
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    #! Vizualizace
    print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.5f}%")

torch.manual_seed(42)
train_time_start_cpu = timer()
epochs = 3

#! Trainig loop
for epochs in tqdm(range(epochs)):
    print(f" Epoch: {epochs}\n================")
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
train_time_end_cpu = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_cpu,
                                            end=train_time_end_cpu,
                                            device=device)
#! Vysledky modelu fromát - Dictionary
def eval_model(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:loss_fn,
               accuracy_fn):
    """ Vrací dictonary s výsledky modelu """
    loss,acc = 0,0
    model_2.eval()
    with torch.inference_mode():
        for X,y in tqdm(data_loader):
            X,y = X.to(device),y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name":model.__class__.__name__,
            "model_loss":loss.item(),
            "model_acc":acc}
model_2_results = eval_model(model=model_2,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)
print(model_2_results)



















