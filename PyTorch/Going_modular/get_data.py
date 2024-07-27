import os
import torch
import zipfile                #! zacal jsem v 9 45
import requests
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_data(target_dir:str,
             data_name:str,
             github_link:str,
             zip_True: bool = True):
    #! """Target dir ->Going_modular/data/"""
    data_path = Path(target_dir)
    image_path = data_path / "pizza_steak_shusi"

    if image_path.is_dir():
        print(f"{image_path} staženo -> přesakuji stahování")
    else:
        print(f"{image_path} není straženo -> stahuji {image_path}")
        image_path.mkdir(parents=True,exist_ok=True)

        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Stahování...")
            f.write(request.content)

            with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip") as zip_ref:
                print("odzipovani")
                zip_ref.extractall(image_path)
            os.remove(data_path / "pizza_steak_sushi.zip")
            