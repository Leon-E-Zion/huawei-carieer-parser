import numpy as np

a = np.random.rand(256,256,3,1)
b = np.random.rand(256,256,3,1)
c = np.random.rand(256,256,3,1)

def ad(a,b,c):
    d = []
    d.append(a)
    d.append(b)
    d.append(c)
    return d

a,b,c = ad(a,b,c)
print(a)





