#https://www.youtube.com/watch?v=omz_NdFgWyU&t=1328s&ab_channel=sentdex
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data
#np.random.seed(0)

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)     # Určí počet neuroný a imputu
        self.biases = np.zeros((1, n_neurons))                              # zaroveň zadani v tomto tvaru nám zaručí že není potřeba použít transpose
        pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1, keepdims=True))
        propabilities = exp_values / np.sum(exp_values,axis=1, keepdims=True )
        self.output = propabilities


X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)       # první číslo v () počet dat které vrstva přijímá a počet a druhé je počet neuronů které chce me vytvořit.
activation1 = Activation_ReLU()        # první imput druhé output

dense2= Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])













