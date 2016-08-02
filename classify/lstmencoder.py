from keras.models import Model
from keras.layers import GRU, Flatten, Reshape, TimeDistributedDense, Dense, Input
from keras import backend as K
from sklearn.cross_validation import train_test_split
import numpy as np
import os
from scipy.spatial.distance import cdist
from keras import regularizers

FNULL = open(os.devnull, 'w')

# import data
ROOT = os.path.dirname(os.path.realpath(__file__))
themes = np.load(ROOT + '/cache/themes128.npz')
themes = themes['arr_0']

# toyise
tdata = []
for i, t in enumerate(themes):
    nt = []
    for j, s in enumerate(t):
        if j % 4 == 0:
            ns = [0, 0, 0, 0]
            # kick
            ns[0] = s[0]
            # sn or stick
            if s[1] > 0:
                ns[1] = s[1]
            elif s[2] > 0:
                ns[1] = s[2]
            else:
                ns[1] = 0.
            ns[2] = s[11]
            if s[12] > 0:
                ns[3] = s[12]
            elif s[14] > 0:
                ns[3] = s[14]
            else:
                ns[3] = 0
            nt.append(ns)
    tdata.append(nt)
tdata = np.array(tdata)
x_train, x_test, _, _ = train_test_split(tdata, tdata, test_size=0.22, random_state=0)
# model
input_dim = 4

inputs = Input(shape=(x_train.shape[1], input_dim))
encoded = GRU(16, activation='relu', return_sequences=True)(inputs)
encoded = TimeDistributedDense(16)(encoded)
encoded = TimeDistributedDense(2)(encoded)
encoded = Flatten()(encoded)
decoded = Dense(128)(encoded)
decoded = Reshape((32,4))(decoded)
decoded = GRU(input_dim, return_sequences=True, activation='hard_sigmoid')(decoded)
m = Model(inputs, decoded)
e = Model(inputs, encoded)

m.compile(optimizer='adadelta', loss='binary_crossentropy')
m.summary()
m.fit(x_train, x_train, nb_epoch=500, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

tt = x_test[0]
rr = np.around(m.predict(np.array([x_test[0]])))[0]
print(x_test[0])
print('======')
print(rr)
print(cdist(tt, rr, 'matching').diagonal().mean())
encoded_themes = e.predict(tdata)
# # # put it in cache
np.savez_compressed(ROOT +'/cache/gru32_themes128.npz', encoded_themes)
