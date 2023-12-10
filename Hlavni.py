#Video https://www.youtube.com/watch?v=tMrbN67U9d4&ab_channel=sentdex   Neural Networks from Scratch - P.3 The Dot Product
#Videohttps://www.youtube.com/watch?v=TEWy9vZcxW4&ab_channel=sentdex
import numpy as np
import matplotlib

np.random.seed(0)

X = [[1, 2, 3, 2.5 ],
    [2.0, 5.0, -1.0, 2.0 ],
    [-1.5, 2.7, 3.3, -0.8]]



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)     # Určí počet neuroný a imputu
        self.biases = np.zeros((1, n_neurons))                              # zaroveň zadani v tomto tvaru nám zaručí že není potřeba použít transpose
        pass
    def forward(self, imputs):
        self.output = np.dot(imputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5)                # zde se zadá počet prvků v imputech (zde je to X), a počet neuronu které chceme vytvořit
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

                                    #print(0.10 * np.random.randn(4, 3)) # vysvětleni v dokumetaci





















