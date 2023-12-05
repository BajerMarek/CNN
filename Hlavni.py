import numpy as np
import matplotlib
imputs = [1, 2, 3, 2.5 ]

weights = [[ 0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87 ]]
biases = [2, 3, 0.5]

layer_outputs = [] # vysledky dane vrstvy
for neuron_weights, neuron_bias in zip(weights, biases): # zip da dohromady dva lysty [0] = weights[0], biases[0]
    neuron_output = 0 # výsledek danehoneuronu
    for n_input, weight in zip(imputs, neuron_weights): # da dohromady imputy a vahu neuronu
        neuron_output += n_input*weight # vytvoři vysledek neuronu
    neuron_output += neuron_bias # přičte bias k vysledku neuronu
    layer_outputs.append(neuron_output)
print(layer_outputs)