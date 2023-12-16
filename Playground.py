import matplotlib.pyplot as plt
import nnfs 
from nnfs.datasets import vertical_data
import numpy as np

nnfs.init

X, y = vertical_data(samples=100, classes=3)

plt.scatter(X[:,0], X[:,1], c=y,s=40,cmap="brg" )
plt.show()

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

class Loss:
    def calculate(self, output, y ):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoriCalcrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correcr_confidence = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 1:
            correcr_confidence = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correcr_confidence)
        return negative_log_likelihoods

dense1 = Layer_Dense(2,3)     
activation1 = Activation_ReLU()     
dense2= Layer_Dense(3,3)
activation2 = Activation_Softmax()

loss_function= Loss_CategoriCalcrossEntropy()

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(100000):
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)

    if loss < lowest_loss:
        print("New set of weights found, iteration:", iteration, "loss:", loss, "acc:", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss