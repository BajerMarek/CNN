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
device = "cuda" if torch.cuda.is_available() else "cpu"

#! DATA 
#todo Část z Food101
#_ náš data set bude mít 3 classy po  75 pro trainig a 25 pro testing
#_ je důležité začínat s málem a níásledně pokračovat s věším
#_ důležiré pro zrychlení celého procesu

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

walk_through_dir(image_path)

#! Rozdělení dat na traing a testing části
train_dir = image_path / "train"
test_dir = image_path / "test"

#! VIZUALIZACE - náhodné fotky z datasetu
#todo Jak se to provede
#_ získáme cesty daných fotek
#_ vybereme náhodnou fotku pomocí random.choice()
#_ získáme jména složek pomocí pathlib.Path.parent.stem
#_ otevřeme fotky pomocí PIL
#_ tobrazíme fotku a metadata

random.seed(42)

#? získáme cety daných fotek
image_path_list = list(image_path.glob("*/*/*.jpg")) #? glob dá všechny cety dohromady * - cokoly
#print(image_path_list)

#? vybereme náhodnou fotku pomocí random.choice()
random_image_path = random.choice(image_path_list)
print(random_image_path)

#? získání class fotky
image_class = random_image_path.parent.stem #? stem = class dané fotky
print(image_class)

#? otevýrání fotky
img = Image.open(random_image_path)

#? zobrazení metadat
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")

#? pomocí matplotlib
img_array = np.array(img)
plt.figure(figsize=(10,7))
plt.imshow(img_array)
plt.title(f"Image class: {image_class} | Image shape: {img_array.shape} -> [height, width, color_channels] [HWC]")
plt.axis("off")
plt.show()

#! Transfrmování dat
#todo Postup
#_ 1. Konvertování dat na PyTorch tensory - číselné znázornění fotky
#_ 2. Převedení na torch.utils.data.Dataset => Dataset formát
#_                 torch.utils.data.DataLoader => DataLoader formát 

#! Transform
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#? transform pro fotku
data_transform = transforms.Compose([
    #? změna velikosti fotky
    transforms.Resize(size=(64,64)),
    #? otočení fotek horizontálně
    transforms.RandomHorizontalFlip(p=0.5),
    #? převedení fotky na tensor
    transforms.ToTensor()
])
print(data_transform(img).shape)

def plot_transformed_images(image_paths, transform,n=3, seed=42):
    """
    Vybere náhodné fotky a po transformaci je zobrazí
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths,k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis("off")


        transformed_image = transform(f).permute(1,2,0)
        ax[1].imshow(transformed_image)
        ax[1].set_title(f"Transformovaná fotka\nShape: {transformed_image.shape}")
        ax[1].axis("off")

        fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
        plt.show()

plot_transformed_images(image_path_list,
                        transform=data_transform,
                        n=3,
                        seed=42)

#! Nahrávání fote pomocí funkce ImageFolder
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,         #? přemnění dat
                                  target_transform=None)        #? pemnění nazve dat
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)
print(train_data)
print(test_data)

#? získávání class jmeno dat jako list
class_names =train_data.classes
print(class_names)

#? získávání class jmeno dat jako dict
class_dict = train_data.class_to_idx
print(class_dict)

#? zišťování velikosti datasetu
print(len(train_data))
print(len(test_data))
print(train_data.samples[0])

#! Vizualizace dat z našeho datasetu
#? můžeme dát učitému indexu daný název a fotku
img, label = train_data[0][0],train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Lable datatype: {type(label)}")
#! matplotlib
#? zmně pozic v listu
img_permute = img.permute(1,2,0)

print(f"Original shape: {img.shape} -> [color_channels, height,width]")
print(f"Image permute: {img_permute.shape} -> [height,width,color_channels]")

#? zobrazení fotky
plt.figure(figsize=(10,7))
plt.imshow(img_permute)
plt.axis("off")
plt.title(class_names[label], fontsize=16)
plt.show()