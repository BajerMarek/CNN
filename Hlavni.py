#Video https://www.youtube.com/watch?v=tMrbN67U9d4&ab_channel=sentdex   Neural Networks from Scratch - P.3 The Dot Product
import numpy as np
import matplotlib
imputs = [1, 2, 3, 2.5 ]
weights = [[ 0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87 ]]
biases = [2, 3, 0.5]


output = np.dot(weights, imputs)+ biases # weights musí být vždy v zavorce první protože určuje počet neuronů - 3 listy 
print(output)
























