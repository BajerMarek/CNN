#Video https://www.youtube.com/watch?v=tMrbN67U9d4&ab_channel=sentdex   Neural Networks from Scratch - P.3 The Dot Product
#Videohttps://www.youtube.com/watch?v=TEWy9vZcxW4&ab_channel=sentdex
import numpy as np
import matplotlib
imputs = [[1, 2, 3, 2.5 ],
          [2.0, 5.0, -1.0, 2.0 ],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[ 0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87 ]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs1 = np.dot(imputs, np.array(weights).T)+ biases
layer1_outputs2 = np.dot(layer1_outputs1, np.array(weights2).T)+ biases2
print(layer1_outputs2)
























