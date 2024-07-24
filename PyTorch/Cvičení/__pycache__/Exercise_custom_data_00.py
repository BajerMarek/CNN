# Import torch
import torch
from torch import nn
from pathlib import Path
import requests
import zipfile
import matplotlib.pyplot as plt
# Exercises require PyTorch > 1.10.0
#print(torch.__version__)       #? zobrazí verzy pytorch

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/")
image_path = data_path / "Cviceni_custom_00"

if image_path.is_dir():
    print(f"{image_path} data jsou uz stažena -> nestahuji")
else:
    print(f"{image_path} data nejsou stažena -> stahuji")
    image_path.mkdir(parents=True,exist_ok=True)        #? udělá složku

    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:  #? otevře sloýku na dané místš
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")    #? stáhne zip z githubu
        print("Downloading...")
        f.write(request.content)    #? vloží zip do složky

    
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip","r") as zip_ref:   #? otevře zip na daném místě
        print("Odzipování")
        zip_ref.extractall(image_path)      #? extrahuje zip 

# 2. Become one with the data
import os
def walk_through_dir(dir_path):
  """Walks through dir_path returning file counts of its contents."""
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(image_path)


# Setup train and testing paths
train_dir = image_path / "train"    #? udělá složku train
test_dir = image_path / "test"      #? udělá složku test

# Visualize an image
image_path_list = list(image_path.glob("*/*/*.jpg"))
print(image_path_list[0])

# Do the image visualization with matplotlib
plt.imshow(image_path_list[0])
plt.show()