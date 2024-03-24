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
#_ získáme cety daných fotek
#_ vybereme náhodnou fotku pomocí random.choice()
#_ získáme jména složek pomocí pathlib.Path.parent.stem
#_ otevřeme fotky pomocí PIL
#_ tobrazíme fotku a metadata

random.seed(42)

#? získáme cety daných fotek
image_path_list = list(image_path.glob("*/*/*.jpg")) #? glob dá všechny cety dohromady * - cokoly
print(image_path_list)

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