
import matplotlib as plt
import numpy as np

def f (x) :
    return 2*x**2

x = np.array(range(5))
y = f(x)

p2_delta = 0.0001

x1 = 1
x2 = x1+p2_delta

y1 = f(x1)
y2 = f(x2)

approcimare_derivative = (y2 - y1) / (x2 - x1)
print(approcimare_derivative)
print(x)
print(y)
print((y[1] - y[0] / (x[1] - x[0])))




