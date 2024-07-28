import os
import torch
import zipfile                #! zacal jsem v 9 45
import requests
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_data_github(target_dir:str,
             data_name:str,
             github_link:str,
             zip_True: bool = True):
    #! """Target dir ->Going_modular/data/"""
    data_path = Path(target_dir)
    image_path = data_path / data_name

    if image_path.is_dir():
        print(f"{image_path} staženo -> přesakuji stahování")
    else:
        print(f"{image_path} není straženo -> stahuji {image_path}")
        image_path.mkdir(parents=True,exist_ok=True)

        if zip_True == True:
            with open(data_path / data_name+"zip", "wb") as f:
                request = requests.get(github_link)
                print("Stahování...")
                f.write(request.content)

                with zipfile.ZipFile(data_path / data_name+"zip") as zip_ref:
                    print("odzipovani")
                    zip_ref.extractall(image_path)
                os.remove(data_path / data_name+"zip")
        else:
            with open(data_path / data_name+"zip", "wb") as f:
                request = requests.get(github_link)
                print("Stahování...")
                f.write(request.content)
              
            