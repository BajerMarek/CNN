
#! CÍL:
#todo Zreplikovat funkci ImageFolder ->
#_ 1. náhrát forku ze složky
#_ 2. získat jeméno fotky z dtasetu
#_ 3. získat classy jako dicty z dtasetu

#? Výhody
#_ Schpnost udělat dataset skoro ze všeho
#_ Nejí žádná limitace od PyTorch

#? Nevýhody
#_ I když jde udělat dataset skoro ze veho tak to neznaméná že to bude fungovat
#_ nutnost psát výce kódu -> výce možností udělat error
#! začíná na řádku 155
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

#walk_through_dir(image_path)

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
#print(random_image_path)

#? získání class fotky
image_class = random_image_path.parent.stem #? stem = class dané fotky
#print(image_class)

#? otevýrání fotky
img = Image.open(random_image_path)

#? zobrazení metadat
#print(f"Random image path: {random_image_path}")
#print(f"Image class: {image_class}")
#print(f"Image height: {img.height}")
#print(f"Image width: {img.width}")

#? pomocí matplotlib
img_array = np.array(img)
plt.figure(figsize=(10,7))
plt.imshow(img_array)
plt.title(f"Image class: {image_class} | Image shape: {img_array.shape} -> [height, width, color_channels] [HWC]")
plt.axis("off")
#plt.show()

#! Transfrmování dat
#todo Postup
#_ 1. Konvertování dat na PyTorch tensory - číselné znázornění fotky
#_ 2. Převedení na torch.utils.data.Dataset => Dataset formát
#_                 torch.utils.data.DataLoader => DataLoader formát 

#! Transform
import torch.utils
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
#print(data_transform(img).shape)

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

#plot_transformed_images(image_path_list,
#                        transform=data_transform,
#                        n=3,
#                        seed=42)
#! Začátek
import os
import pathlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

#! Získávání class names
#todo Co je cílem:
#_ 1. za použití os.scandir() získat jmeno class ( idealně v standartním formátu)
#_ 2. vyhodit error když nenajdeme žádná jména -> špatne formát složek (chyby v normalizaci)
#_ 3. vzít jéna class a udělat z nich list a dict a zobrazit je

target_directory = train_dir
print(f"Target dir: {target_directory}")

class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
print(class_names_found)
print(list(os.scandir(target_directory)))

#! funkcionalizace

def find_classes(directory:str) -> Tuple[List[str], Dict[str,int]]:         #? to ce je za "->" znároňuje co bude outputem dané funkce
    """
    Najde jména class v daném místě
    """
    #? 1, získání jmen skenováním danéko dir
    classes = sorted(entry.name for entry in os.scandir(directory)if entry.is_dir())

    #? 2. vyhodit error když nenajdeme žádná jména
    if not classes:
        raise FileNotFoundError(f" Nebyly nalezeny žádná jména class v {directory}... prosím skontrolujte složky")
    
    #? 3. vzít jéna class a udělat z nich list a dict -> počítače mají raději čísla než písmena jako label
    class_to_idx = {class_name:i for i, class_name in enumerate(classes)}

    return classes,class_to_idx

print(find_classes(target_directory))

#! Nahrávání fote pomocí funkce ImageFolder
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,         #? přemnění dat
                                  target_transform=None)        #? pemnění nazve dat
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)
class_names =train_data.classes
class_dict = train_data.class_to_idx

#! Vytvoření  vlastního "Datasetu"
#todo Co bude potřeba:
#_ 1. subclass torch.utils.data.Dataset
#_ 2, init subclass a transformovat data
#_ 3. Vytvořit určité atributy:     (pro fotky)
#_      paths -  odkud jsou naše fotky
#_      transform - transformoavat data
#_      classes - list požadovaných class
#_      class:to:idx - index daných class
#_ 4. udělat funkci load_image() -> otevře fotku
#_ 5. přepsat __len()__ pro správné obrazení délky datasetu
#_ 6. přepsat __getitem()__ zobrazý fotku s daným indexem

from torch.utils.data import Dataset
#? 1. subclass
class ImageFolderCustom(Dataset):
    #? 2. inicializace
    def __init__(self,
                 targ_dir:str,
                 transform=None) -> None:
        #? vytvoření atrybutů
        #? paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))           #? list všech path které vypadají -> */*/.jpg v daném místě
        #? transform
        self.transform = transform
        #? class_to_idx
        self.classes, self.class_to_idx = find_classes(targ_dir)
        #? load image
    def load_image(self,index:int) -> Image.Image:
        """
        Otevře fotku na místě určením path
        """
        image_path= self.paths[index]
        return Image.open(image_path)
        #? pepsání len
    def __len__(self) -> int:
        """Vrátí celkový počet dat"""
        return len(self.paths)                              
    def __getitem__(self, index:int)-> Tuple[torch.Tensor, int]:
        """Vrací jedenu ukázku dat"""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name              #? předpokládaný formát: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]
    
        #? pokud nutné transformování
        if self.transform:
            return self.transform(img),class_idx                #? vrátí fotku jako tensor a index dané classy
        else:
            return img, class_idx                               #? vrací bez transformace

#? vytvoření transformu
from torchvision import transforms
train_transform = transforms.Compose([
                                    transforms.Resize(size=(64,64)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()
])

test_transform = transforms.Compose([
                                    transforms.Resize(size=(64,64)),
                                    transforms.ToTensor()
])

#! Test ImageFolderCustom
train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                      transform=train_transform)

test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                     transform=test_transform)
#print(train_data_custom.classes)
#print("-------------------------------------------------------------------------------------------------------")
#print(test_data_custom.class_to_idx)
#print("-------------------------------------------------------------------------------------------------------")

#! Vizualizace
#todo Funkce na zobrazování náhodných fotek z datasetu:
#_ 1. Ziskat dataset a informace o datech z něj
#_ 2. Zobrazení maximálně 10 fotek
#_ 3. Nastavení seedu
#_ 4. Získat list náhodných idexu dat (fotek)
#_ 5. Mathplot lib
#_ 6. Porojet jednotlyvé fotky azobrazit je pomocí mathplotlib
#_ 7. Přehodi dimenze tak aby seděli s mathplotlib

#? Získání dat
def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape : bool = True,
                          seed : int = 42):
    #? omezení 10
    if n>10:
        n = 10
        display_shape=False
        print(f"Kvůli zobrazení n zredukováno na 10 a display_shape nataven na False")

    #? Seed
    if seed:
        random.seed(seed)
    #? Náhodná data
    random_sample_idx = random.sample(range(len(dataset)), k=n)
    
    #? Mathplotlib
    plt.figure(figsize=(16,8))

    #? Projetí dat
    for i , targ_sampel in enumerate(random_sample_idx):
        targ_image, targ_label = dataset[targ_sampel][0], dataset[targ_sampel][1]

        #? Dimenze
        targ_image_adjust = targ_image.permute(1,2,0)   #? [color_channels, height,width] -> [height,width,color_channels]

        #? zobrazení upravené fotky
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nShape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()

"""
display_random_images(train_data_custom,
                      n=12,
                      classes=class_names,
                      seed=None)
"""
#! Datasety
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
train_dataloader_custom = DataLoader(dataset=train_data_custom,
                               batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS,
                               shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                               batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS,
                               shuffle=False)

#img_custom, label_custom = next(iter(train_dataloader_custom))     #! dělá problémy
#print(img_custom.shape)
#print("----------------")
#print(label_custom.shape)

#! Augmentace dat
#todo O co se jedná:
#_ Schválné zpestření datasetu
#_  v případě fotem se může jednat o různé otáčení zmněnu barev ...

train_transform =  transforms.Compose([
                                    transforms.Resize(size=(224,224)),
                                    transforms.TrivialAugmentWide(num_magnitude_bins=5),
                                    transforms.ToTensor()
])
test_transform = transforms.Compose([
                                    transforms.Resize(size=(224,224)),
                                    transforms.TrivialAugmentWide(num_magnitude_bins=5),
                                    transforms.ToTensor()
])
image_path_list_custom = list(image_path.glob("*/*/*.jpg"))
image_path_list_custom[:10]
plot_transformed_images(image_paths=image_path_list_custom,
                        transform=train_transform,
                        n=6,
                        seed=None)