import numpy as np
import nnfs

nnfs.init()

layer_otputs = [[4.8, 1.2, 2.385],
                [8.9, -1.881, 0.2],
                [1.41, 1.051, 0.026]]


# exponencializace
exp_values = np.exp(layer_otputs)   # převede output na exponent E = eulerovo  číslo

# normalizace 
norm_values = exp_values / np.sum(exp_values,axis=1, keepdims=True)   # Tím se získá pravděpodobnost do promněné norm_values

print(norm_values)
#print(sum(norm_values))                     # Suma pravděpodobností je vždy 1
