import numpy as np
xy = np.mgrid[-5:5.1:0.5, -5:5.1:0.5].reshape(2,-1).T

print(xy)