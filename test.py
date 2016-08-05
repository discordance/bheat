import numpy as np
import utils
# xy = np.mgrid[-5:5.1:0.5, -5:5.1:0.5].reshape(2,-1).T
#
# print(xy)

test = np.array([
    [1,0,0,0, 0.3, 0.0],
    [0,0,0,0, 0.0, 0.0],
    [0,0,1,0, 0.0, 0.5],
    [1,0,0,0, 0.5, 0.0],
    [1,0,0,1, 0.2, 0.2],
])

vels = test[:,-2:]

for i, vel in enumerate(vels.T):
    med = vel[np.nonzero(vel)].mean()
    gaps = med-vel[np.nonzero(vel)]
    vel[np.nonzero(vel)] = vel[np.nonzero(vel)]+(gaps/1.3)

trunk = np.delete(test, np.s_[4::], 1)

gap = 0.85-np.max(vels)
vels[np.nonzero(vels)] = vels[np.nonzero(vels)]+gap
trunk = np.column_stack((trunk, vels))
print(trunk)