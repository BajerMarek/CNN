import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch
#!prestavka 11 25
class TinyVgg(nn.Module):
    """Replika modelu TinyVgg z webu CNN explainer -> https://poloclub.github.io/cnn-explainer/"""
    def __init__(self,
                 input_shape:int,
                 hidden_units:int,
                 output_shape:int)->None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,         #! tady bude error
                      out_features=output_shape)
        )
    def forward(self,x:torch.Tensor):
        x = self.block1(x)  
        x = self.block2(x)
        x = self.classifier(x)
        return x
    