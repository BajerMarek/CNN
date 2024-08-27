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

from torchvision.transforms import Resize, InterpolationMode

#! vrstvy modelu
class ConvBNSiLU(nn.Module):
    def __init__(self,in_channels, out_channel, kernel_size,stride,padding):
        super(ConvBNSiLU,self).__init__()

        self.cbl = nn.Sequential(
            #? Conv čast funkce(ConvBNSiLU)
            nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channel,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=False),
            #? BN čast funkce(ConvBNSiLU)
            nn.BatchNorm2d(num_features=out_channel,
                           eps=1e-3,
                           momentum=0.03),
            nn.SiLU(inplace=True)
        )
    def forward(self,x):
        return self.cbl(x)
    
class BotttleNeck(nn.Module):
    """Parametry:
        channels - dimenze ve které sou uloženy barvy fotky dat [Height,Width,(color)channels]
        in_channels -> počet channels vstupního tensoru         
        out_channels -> počet channels výstupního tensoru
        width_multiple -> řídí počet channels a weights u všech konvolucí kromně první a poslední
                          Pokud bíže nule -> modle je jednoduší. pokud blíže 1 -> model je složitější"""
    
    def __init__(self,in_channels:int,out_channels:int,width_multiple=1):
        super(BotttleNeck, self).__init__()
        channels = int(width_multiple*in_channels)
        self.c1 = ConvBNSiLU(in_channels=in_channels,out_channel=channels,kernel_size=1,stride=1, padding=0)
        self.c2 = ConvBNSiLU(in_channels=channels,out_channel=out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        return self.c2(self.c1(x)) + x
    
class C3(nn.Module):
    """Parametry:
    channels - dimenze ve které sou uloženy barvy fotky dat [Height,Width,(color)channels]
    in_channels -> počet channels vstupního tensoru         
    out_channels -> počet channels výstupního tensoru
    width_multiple[float] -> řídí počet channels a weights u všech konvolucí kromně první a poslední
                      Pokud bíže nule -> modle je jednoduší. pokud blíže 1 -> model je složitější
    depth[int] -> určuje kolikrát bude BottleNeck aplikován v C3 bloku
    backbone[bool] -> pokud True self.seq bude BottleNeck1
                Pokud False bude použit BottleNeck2 """
    def __init__(self,in_channels:int,
                 out_channels:int,
                 width_multiple=1,
                 depth=1,
                 backbone=True):
        super(C3,self).__init__()

        channels = int(width_multiple*in_channels)

        self.c1 = ConvBNSiLU(in_channels=in_channels,out_channel=channels,kernel_size=1,stride=1,padding=0)
        self.c_skipped = ConvBNSiLU(in_channels=in_channels,out_channel=channels,kernel_size=1,stride=1,padding=0)

        if backbone:
            self.seq = nn.Sequential(*[BotttleNeck(in_channels=channels,out_channels=channels,width_multiple=1)for _ in range(depth)])
        else:
            self.seq = nn.Sequential(*[nn.Sequential(ConvBNSiLU(in_channels=channels,out_channel=channels,kernel_size=1,stride=1,padding=0),
                                                     ConvBNSiLU(in_channels=channels,out_channel=channels,kernel_size=3,stride=1,padding=1))for _ in range(depth)])
            
        self.c_out = ConvBNSiLU(in_channels=channels*2,out_channel=out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = torch.cat(self.seq(self.c1(x)),self.c_skipped(x),dim=1)
        return self.c_out(x)
    
class SPPF(nn.Module):
    def __init__(self,in_channels:int,out_channels):
        super(SPPF,self).__init__()

        channels = int(in_channels/2)

        self.c1 = ConvBNSiLU(in_channels=in_channels, out_channel=out_channels,kernel_size=1,stride=1,padding=0)
        self.pool = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.c_out = ConvBNSiLU(in_channels=channels*4,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)
        return self.c_out(torch.cat([x,pool1,pool2,pool3], dim=1))
    
class HEADS(nn.Module):
    def __init__(self,
                 nc=80,
                 anchors=(),
                 ch=()):
        super(HEADS,self).__init__()
        self.nc= nc                             #? počet class
        self.nl = len(anchors)                  #? počet detekčních vrstev
        self.naxis = len(anchors[0])            #? počet kotev na predikci (predikci na jedne dimenzi ne na celé fotce)
        self.stride = [8,16,32]

        anchors_ = torch.tensor(anchors).float().view(self.nl, -1,2) / torch.tensor(self.stride).repeat(6,1).T.reshape(3,3,2)
        self.register_buffer('anchors',anchors_)

        self.out_convs = nn.ModuleList()
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels,out_channels=(5+self.nc)*self.naxis,kernel_size=1)
            ]
        
    def forward(self,x):
        for i in range(self.nl):
            x[i] = self.out_convs[i](x[i])
            bs,_,grid_y,grid_x = x[i].shape
            x[i] = x[i].view(bs,self.naxis,(5+self.nc), grid_y,grid_x).permute(0,1,3,4,2).contiguous()

        return x

#! model
class JOLOv5m(nn.Module):
    def __init__(self,first_out,nc=80,anchors=(),ch=(),infrence = False):
        super(JOLOv5m,self).__init__()
        self.infrence = infrence
        self.backbone = nn.ModuleList()
        self.backbone +=[
            ConvBNSiLU(in_channels=3,out_channel=first_out,kernel_size=6,stride=2,padding=2),
            ConvBNSiLU(in_channels=first_out,out_channel=first_out*2,kernel_size=3,stride=2,padding=1),
            C3(in_channels=first_out*2,out_channels=first_out*2,width_multiple=0.5,depth=2),
            ConvBNSiLU(in_channels=first_out*2,out_channel=first_out*4,kernel_size=3,stride=2,padding=1),
            C3(in_channels=first_out*4,out_channels=first_out*4,width_multiple=0.5,depth=4),
            ConvBNSiLU(in_channels=first_out*4, out_channel=first_out*8,kernel_size=3,stride=2,padding=1),
            C3(in_channels=first_out*8,out_channels=first_out*8,width_multiple=0.5,depth=6),
            ConvBNSiLU(in_channels=first_out*8,out_channel=first_out*16,kernel_size=3,stride=2,padding=1),
            C3(in_channels=first_out*16,out_channels=first_out*16,width_multiple=0.5,depth=2),
            SPPF(in_channels=first_out*16,out_channels=first_out*16)
        ]
        self.neck = nn.ModuleList()
        self.neck += [
            ConvBNSiLU(in_channels=first_out*16,out_channel=first_out*16,kernel_size=1,stride=1,padding=0),
            C3(in_channels=first_out*16,out_channels=first_out*8,width_multiple=0.25,depth=2,backbone=False),
            ConvBNSiLU(in_channels=first_out*8,out_channel=first_out*8,kernel_size=1,stride=1,padding=0),
            C3(in_channels=first_out*8,out_channels=first_out*4,width_multiple=0.25,depth=2,backbone=False),
            ConvBNSiLU(in_channels=first_out*4,out_channel=first_out*4,kernel_size=3,stride=2,padding=1),
            C3(in_channels=first_out*8,out_channels=first_out*8,width_multiple=0.5,depth=2,backbone=False),
            ConvBNSiLU(in_channels=first_out*8,out_channel=first_out*8,kernel_size=3,stride=2,padding=1),
            C3(in_channels=first_out*16,out_channels=first_out*16,width_multiple=0.5,depth=2,backbone=False)
        ]
        self.head = HEADS(nc=nc,anchors=anchors,ch=ch)
    def forward(self,x):
        assert x.sahpe[2] % 32 == 0 and x.shape[3] % 32 ==0, "Výška a šířka jsou nedělitelné"
        bacbone_connection =[]
        nec_connection = []
        outputs = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in[4,6]:
                #? získá hodnoty z drehého a třetího C3 bloku a uloží je
                bacbone_connection.append(x)
            
        for idx, layer in enumerate(self.neck):
            if idx in [0,2]:
                x = layer(x)
                nec_connection.append(x)
                x = Resize([x.shape[2]*2,x.shape[3]*2],interpolation=InterpolationMode.NEAREST)(x)
                x = torch.cat([x,bacbone_connection.pop(-1)],dim=1)
            
            elif idx in [4,6]:
                x = layer(x)
                x = torch.cat([x,nec_connection.pop(-1)],dim=1)
            elif isinstance(layer,C3) and idx >2:
                x = layer(x)
                outputs.append(x)
            
            else:
                x = layer(x)
            
        return  self.head(outputs)