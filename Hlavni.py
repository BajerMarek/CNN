#https://www.youtube.com/watch?v=gmjzbpSVY1A&ab_channel=sentdex
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
#np.random.seed(0)

X = [[1, 2, 3, 2.5 ],
    [2.0, 5.0, -1.0, 2.0 ],
    [-1.5, 2.7, 3.3, -0.8]]
X, y = spiral_data(100,3)
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

layer1 = Layer_Dense(2,5)                # zde se zadá počet prvků v imputech (zde je to X), a počet neuronu které chceme vytvořit
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)







#print(0.10 * np.random.randn(4, 3)) # vysvětleni v dokumetaci
"""
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
print(output)
"""

"""
# data set pro CNN => nahodný data.
#https://cs231n.github.io/neural-networks-case-study/
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y
"""




















