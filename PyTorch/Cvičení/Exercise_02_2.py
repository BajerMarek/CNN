import numpy as np
import matplotlib.pyplot as plt

def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

x_values = np.linspace(-5, 5, 100)  

y_values = tanh(x_values)

plt.plot(x_values, y_values, label='y = x^2 + 2x + 1')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of a Quadratic Function')
plt.legend()
plt.show()
