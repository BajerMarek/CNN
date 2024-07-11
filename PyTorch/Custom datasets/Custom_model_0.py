if __name__ == '__main__':
    import torch
    from torch import nn
    #? Data
    import requests
    import zipfile
    from pathlib import Path
    import os
    #? vizualizace - fotek
    from PIL import Image
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torchvision import datasets
    #! DtaLoader
    from torch.utils.data import DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #? připaví cestu pro data
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    #? pokud slozku pro data uz nemáme vytvoří se
    if image_path.is_dir():
        print(f"{image_path} data uz máš... přeskakuji stahování")
    else:
        print(f"{image_path} data neexistují ... stahuji")
        image_path.mkdir(parents=True,exist_ok=True)

    #? stahování dat
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading...")
        f.write(request.content)

        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip","r") as zip_ref:
            print("Odzipování")
            zip_ref.extractall(image_path)


    #! DATA - pokus porozumnět jim

    def walk_through_dir(dir_path):
        """ Projde dir_path a vrátí to co obsahuje"""
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"Tady je {len(dirnames)} složek a {len(filenames)} fotek v '{dirpath}'.")

    #! Rozdělení dat na traing a testing části
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    #! Pro vizualizaci
    #? získáme cety daných fotek
    image_path_list = list(image_path.glob("*/*/*.jpg")) #? glob dá všechny cety dohromady * - cokoly
    #? vybereme náhodnou fotku pomocí random.choice()
    random_image_path = random.choice(image_path_list)
    #? získání class fotky
    image_class = random_image_path.parent.stem #? stem = class dané fotky

    #! Model TinyVGG
    #todo Postup:
    #_ 1. Transformace

    #! Transformace dat
    simple_transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    ])

    train_data_simple = datasets.ImageFolder(root=train_dir,
                                      transform=simple_transform,
                                      target_transform=None)
    test_data_simple = datasets.ImageFolder(root=test_dir,
                                     transform=simple_transform)

    BATCH_SIZE = 32
    WORKERS= os.cpu_count()
    train_dataloader_simple = DataLoader(dataset=train_data_simple,
                                  batch_size=BATCH_SIZE,
                                  num_workers=WORKERS,
                                  shuffle=True)
    test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                 batch_size=BATCH_SIZE,
                                 num_workers=WORKERS,
                                 shuffle=False)
    class_names =train_data_simple.classes
    class_dict = train_data_simple.class_to_idx
    #! Baseline model
    #! Architecture: conv62 - relu -> conv60 - relu60 - maxpool30 -> conv28 - relu28 -> conv26 - relu26 -maxpool13

    class TinyModelCustom(nn.Module):
        def __init__(self, input_shape:int,hidden_units:int,output_shape:int):
            """
            Kopie modelu TinyVGG 
            """
            super().__init__()
            self.block1=nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU()
            )
            self.block2=nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                            stride=2)
            )
            self.block3=nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=0),
                nn.ReLU()
            )
            self.block4=nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                            stride=2)
            )
            self.classifier=nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*13*13,
                        out_features=output_shape)
            )
        def forward(self,x):
            x = self.block1(x)
            print(f"Shape of x: {x.shape}")
            x = self.block2(x)
            print(f"Shape of x: {x.shape}")
            x = self.block3(x)
            print(f"Shape of x: {x.shape}")
            x = self.block4(x)
            print(f"Shape of x: {x.shape}")
            x = self.classifier(x)
            return x
    model_0 = TinyModelCustom(input_shape=3,
                    hidden_units=10,
                    output_shape=len(class_names)).to(device)
    image_batch,label_batch = next(iter(train_dataloader_simple))
    print(model_0(image_batch))