import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List
from PIL import Image
import shutil
složka = "Kostky_dataset"
složka_dat = "C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data"
path = os.path.join(složka_dat,složka)
#os.mkdir(path)
data_path = 'C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\Kostky\\features_npy.npy'
label_path = 'C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\Kostky\\labels_npy.npy'
data = np.load(data_path)
label =np.load(label_path)
print(f"Shape of the data: {data.shape}")
print(f"Data type: {data.dtype}")

print(data[0])
print("##################################################################")
print(f"Shape of the label: {label.shape}")
print(f"Data type: {label.dtype}")
print(label)
print(data.shape[0]/100*20)

for i in range(480000):
    file_name = f"image_{i}.jpg"
    directory ="C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\"
    if i <= 24000:
        train = Image.fromarray(data[i])
        local_file =train.save(f"{directory}{file_name}")
        destination_dir ="C:\\Users\\Gamer\Desktop\\111\\Programování\\CNN\data\\Kostky_dataset\\train\\CUBE\\"
        destination_full = os.path.join(destination_dir,local_file)
        shutil.move(data_path,destination_full)

    if i>24000 and i<=120000:
        train = Image.fromarray(data[i])
        train.save(f"image_{i}.jpg")
        destination_dir ="C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\Kostky_dataset\\test\\CUBE\\"
        destination_full = os.path.join(destination_dir,f"image_{i}.jpg")
        shutil.move(data_path,destination_full)

    if i <= 164000:
        train = Image.fromarray(data[i])
        train.save(f"image_{i}.jpg")
        destination_dir ="C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\Kostky_dataset\\train\\CONE\\"
        destination_full = os.path.join(destination_dir,f"image_{i}.jpg")
        shutil.move(data_path,destination_full)
        
    if i>24000 and i<=240000:
        train = Image.fromarray(data[i])
        train.save(f"image_{i}.jpg")
        destination_dir ="C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\Kostky_dataset\\test\\CONE\\"
        destination_full = os.path.join(destination_dir,f"image_{i}.jpg")
        shutil.move(data_path,destination_full)

    if i <= 26400:
        train = Image.fromarray(data[i])
        train.save(f"image_{i}.jpg")
        destination_dir ="C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\Kostky_dataset\\train\\SPHERE\\"
        destination_full = os.path.join(destination_dir,f"image_{i}.jpg")
        shutil.move(data_path,destination_full)
        
    if i>24000 and i<=360000:
        train = Image.fromarray(data[i])
        train.save(f"image_{i}.jpg")
        destination_dir ="C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\Kostky_dataset\\test\\SPHERE\\"
        destination_full = os.path.join(destination_dir,f"image_{i}.jpg")
        shutil.move(data_path,destination_full)

    if i <= 384000:
        train = Image.fromarray(data[i])
        train.save(f"image_{i}.jpg")
        destination_dir ="C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\Kostky_dataset\\train\\TORUS\\"
        destination_full = os.path.join(destination_dir,f"image_{i}.jpg")
        shutil.move(data_path,destination_full)
        
    if i>24000 and i<=480000:
        train = Image.fromarray(data[i])
        train.save(f"image_{i}.jpg")
        destination_dir ="C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\data\\Kostky_dataset\\test\\TORUS\\"
        destination_full = os.path.join(destination_dir,f"image_{i}.jpg")
        shutil.move(data_path,destination_full)

print("hotovo")

plt.imshow(data[0], cmap='gray')
plt.title(label=label[0])
plt.show()