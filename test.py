import numpy as np

A=np.arange(3840).reshape(2,128,15)
coefs = []
for i in range(0, 128):
    if i%32 == 0:
        coefs.append(1.0)
    else:
        coefs.append(0.5)
coefs = np.array(coefs)
print(A)
print('------------')
for i in range(0, coefs.shape[0]):
    A[:,i,:]=A[:,i,:]*coefs[i]
print(A)