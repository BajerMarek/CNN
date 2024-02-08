# Konstrukce CNN
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

#! Čas
def print_train_time(start: float,
                     end: float,
                     device: torch.device =None):
    """ Vypíše rozdíl mezi časem začátku a koncem"""
    total_time = end -start
    print(f"Train time on {device}: {total_time} sekund")
    return total_time

class FashionMNISTModelV3(nn.Module):
    """
    Konstrukce stejná jako pro TinyVGG 
    ze stránky CNN Explainer
    """
    def __init__(self, input_shape: int, hidden_units:int,output_shape:int):
        super().__init__()
        self.conv_blok_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),    #? Hyperparametry
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_blok_2 = nn.Sequential(
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
            nn.Linear(in_features=hidden_units*0,
                      out_features=output_shape)
        )
    def forward(self,x):
        x = self.conv_blok_1(x)
        print(x.shape)
        x = self.conv_blok_2(x)
        print(x.shape)
        x = self.classifier(x)
        return x
    
torch.manual_seed(42)
model_3 = FashionMNISTModelV3(input_shape=1,    #? pocet barev - clour channels
                              hidden_units=10,  #? Pocet neuronu
                              output_shape=len(class_names)).to(device)
print("Bez chyb")