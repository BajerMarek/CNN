import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

field_path_00 = "data/model_df/model_0_df"
field_path_01 = "data/model_df/model_01_df"
field_path_02 = "data/model_df/model_02_df"
field_path_cviceni = "data/model_df/model_cviceni_df"
model_0_df = pd.read_csv(field_path_00)
model_01_df = pd.read_csv(field_path_01)
model_02_df = pd.read_csv(field_path_02)
model_cviceni_df = pd.read_csv(field_path_cviceni)
#print(model_0_df)

    #! porovnávání výsledků modelů -> VIZUALIZACE
plt.figure(figsize=(15,30))
epochs = range(len(model_0_df))
#? train loss
plt.subplot(2,2,1)
plt.plot(epochs, model_0_df["train_loss"], label = "Model 0")
plt.plot(epochs, model_01_df["train_loss"], label="Model 1")
plt.plot(epochs, model_02_df["train_loss"], label="Model 2")
plt.plot(epochs, model_cviceni_df["train_loss"], label="Model Cv")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()
 
    #? test loss
plt.subplot(2,2,2)
plt.plot(epochs, model_0_df["test_loss"], label = "Model 0")
plt.plot(epochs, model_01_df["test_loss"], label="Model 1")
plt.plot(epochs, model_02_df["test_loss"], label="Model 2")
plt.plot(epochs, model_cviceni_df["test_loss"], label="Model Cv")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

    #? train acc
plt.subplot(2,2,3)
plt.plot(epochs, model_0_df["train_acc"], label = "Model 0")
plt.plot(epochs, model_01_df["train_acc"], label="Model 1")
plt.plot(epochs, model_02_df["train_acc"], label="Model 2")
plt.plot(epochs, model_cviceni_df["train_acc"], label="Model Cv")
plt.title("Train Acc")
plt.xlabel("Epochs")
plt.legend()

    #? test acc
plt.subplot(2,2,4)
plt.plot(epochs, model_0_df["test_acc"], label = "Model 0")
plt.plot(epochs, model_01_df["test_acc"], label="Model 1")
plt.plot(epochs, model_02_df["test_acc"], label="Model 2")
plt.plot(epochs, model_cviceni_df["test_acc"], label="Model Cv")
plt.title("Test Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()