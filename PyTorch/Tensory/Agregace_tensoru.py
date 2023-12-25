import torch
# agregace =  hledání: sumi, min, max, mean...
x = torch.arange(1, 100, 10)
print(x)
print(x.dtype)
#heldání minima
print(torch.min(x), x.min())

# hlední maxima
print(torch.max(x), x.max())

# hledani průmeru
print(torch.mean(x.type(torch.float32)), x.type(torch.float32).mean())    # funke mean funguje pouze s daty o typy float32
                                            # tedy musíme pomoci funkce .type() zmněnit datatyp

# hledani sumi
print(torch.sum(x), x.sum())

# hlední idexu mimimalni hodnoty
print(x.argmin())

# hlední idexu maximani hodnoty
print(x.argmax())