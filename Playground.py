import numpy as np
import matplotlib as plt
np.random.seed(0)

layer_otput = [4.8, 1.2, 2.385]

layer_otput = [4.8, 4.79, 4.25]













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

print("here")
X, y = create_data(100,3)

plt.scatter(X[:,0], X[:.1])
plt.show()

plt.scatter(X[:,0], X[:.1], c=y, camp="brg")

#print(0.10 * np.random.randn(4, 3)) # vysvětleni v dokumetaci

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
print(output)
"""
