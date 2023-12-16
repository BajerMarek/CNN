import math
layer_otput = [4.8, 1.2, 2.385]
#E = 2.71828182846
E = math.e

exp_values = []
# exponencializace

for output in layer_otput:
    exp_values.append(E**output)            # převede output na exponent E = eulerovo  číslo

print(exp_values)
# normalizace 

norm_base = sum (exp_values)
norm_values = []

for value in exp_values:                    # Vydělý output sumou všexh outputů 
    norm_values.append( value/norm_base)    # Tím se získá pravděpodobnost do promněné norm_values

print(norm_values)
print(sum(norm_values))                     # Suma pravděpodobností je vždy 1
