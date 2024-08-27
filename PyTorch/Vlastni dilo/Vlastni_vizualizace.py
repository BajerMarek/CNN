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

def save_and_move(i:int,
                  train_or_test:str,
                  type:str):
    image_array = data[i]
    image = Image.fromarray(image_array)
    directory = 'C:\\Users\\Gamer\\Desktop\\111\\Programování\\CNN\\'
    file_name = f"image_{i}.jpg"
    file_path = os.path.join(directory,file_name)
    image.save(file_path)
    destination_dir =f"C:\\Users\\Gamer\\Desktop\\111\\Programování\\datasety\\Kostky_dataset\\{train_or_test}\\{type}\\"
    shutil.move(file_path,destination_dir)

for i in range(480000):
    file_name = f"image_{i}.jpg"
    if i <= 24000:
        save_and_move(i=i,
                    train_or_test="train",
                    type="CUBE")


    if i>24000 and i<=120000:
        save_and_move(i=i,
                   train_or_test="test",
                   type="CUBE")   

    if i>120000 and i <= 164000:
        save_and_move(i=i,
                    train_or_test="train",
                    type="SPHERE")      
       
    if i>164000 and i<=240000:
        save_and_move(i=i,
                   train_or_test="test",
                   type="SPHERE")        

    if i <= 264000 and i>240000:
        save_and_move(i=i,
                  train_or_test="train",
                  type="CONE")       
   
    if i>264000 and i<=360000:
        save_and_move(i=i,
                train_or_test="test",
                type="CONE")          

    if i <= 384000 and i>360000:
        save_and_move(i=i,
              train_or_test="train",
              type="TORUS")       
    
    if i>384000 and i<=480000:
       save_and_move(i=i,
          train_or_test="test",
          type="TORUS")    

print("hotovo")

plt.imshow(data[0], cmap='gray')
plt.title(label=label[0])
plt.show()