# klasifikace pomocí neuronové sítě
from typing import Self
import torch
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np
import pathlib 
from pathlib import Path
import pandas as pd
# Data
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
# oddělí text caramy
def print2(*args):
    print("=======================================")
    for arg in args:
        print(arg)
    print("=======================================")
# vytvoření kol 1000 (kružnic)
n_samples = 1000
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
print(len(X), len(y))
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")

# Make DataFrae of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "labels": y})
print(circles.head(10))

# Vizualizace - graficky
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
#plt.show()

# 1.1 Převod dat do tensoru
print(f"Tvar X: {X.shape} | Tvar y: {y.shape}")

# Nahled na příklad vzorku vlastností a jmen
X_sampel = X[0]
y_sampel = y[0]
print(f"Hodnota pro X : {X_sampel} to stejné pro y: {y_sampel}")
print(f"Tvar X_sampel: {X_sampel.shape} | Tvar y_sampel: {y_sampel.shape}")

# 1.2 převod + rodělení dat na treniková a na testová
X = torch.from_numpy(X).type(torch.float) # převede s numpy array do torch tensoru
y = torch.from_numpy(y).type(torch.float) # to stejné pro y
print2(X[:5], y[:5])

# Dělení dat
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,   # 0,2 -> 20% jako testova data
                                                    random_state=42)

print2(len(X_train), len(X_test), len(y_train), len(y_test))








