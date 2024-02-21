import torch
from torch import nn
#! Torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
#! Mathplotlib
import matplotlib.pyplot as plt
#! Další
import requests
from pathlib import Path
from timeit import default_timer as timer
from tqdm.auto import tqdm

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
"""
rand_image_tensor = torch.randn(size=(1,28,28))
print(rand_image_tensor.shape)
print(model_4(rand_image_tensor.unsqueeze(0).to(device)))
"""

#! Loss funkce eval metrics optimizer
from helper_functions import accuracy_fn
from Moje_funkce_CV import  train_step,test_step,eval_model

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)
#! funkce
#! Learnig loop
torch.manual_seed(42)
treain_time_start = timer()
epochs = 10
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