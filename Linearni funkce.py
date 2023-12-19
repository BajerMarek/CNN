
import matplotlib.pyplot as plt
import numpy as np

def f (x) :
    return 2*x**2

x = np.array(range(5))
y = f(x)

plt.plot(x, y)

colors = ['k', 'g', 'r', 'b', 'c',]

def approximate_tangent_line(x, approximate_derivate, b) :
    return approximate_derivate*x + b           # y = m*x + b

for i in range (5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)
    print((x1, x2), (x2, x1))
                                                    # b = y -m*x
    approximate_derivate = (y1 - y2) / (x1 - x2)    # b = y2 - approximate_derivate * x2

    b = y2 - approximate_derivate * x2              # vytvoří odhad pomocí přímky kteréá je derivat?


    to_plot = [x1-0.9, x1, x1+0.9]
    plt.scatter(x1, y1, c=colors[i])
    plt.plot(to_plot, [approximate_tangent_line(point, approximate_derivate, b) for point in to_plot])
    print('Aproximate derivate for f(x)', f'where x ={x1} is {approximate_derivate}')

plt.show()
