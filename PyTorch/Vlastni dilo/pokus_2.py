
i =383000
if i <= 24000:
    print("CUBE TRAIN")
if i>24000 and i<=120000:
    print("CUBE TEST")      
if i>120000 and i <= 164000:
    print("SPHERE TRAIN")    
 
if i>164000 and i<=240000:
    print("SPHERE TEST")           
if i <= 264000 and i>240000:
    print("CONE TRAIN")      
 
if i>264000 and i<=360000:
    print("CONE TEST")           
if i <= 384000 and i>360000:
    print("TORUS TEST")    
if i>384000 and i<=480000:
    print("TORUS TRAIN")  