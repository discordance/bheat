from keras.layers import Input, Dense, Dropout, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import os
from sklearn.cross_validation import train_test_split
# local package
import utils
import progressbar
import random

STYLE = 'jazz'
FNULL = open(os.devnull, 'w')
ROOT = os.path.dirname(os.path.realpath(__file__))
# for playback and tests
beats = np.load(ROOT + '/cache/'+STYLE+'.npz')
beats = beats['arr_0']

# transform bars and keep them unique
print('clean beats ...')
nbeats = []
compress = []
bar = progressbar.ProgressBar()

for beat in bar(beats):
    for bar in beat:
        cp = utils.compress(bar)
        if cp not in compress and bar.mean() > 0.0125:
            nbeats.append(bar)
            compress.append(cp)
beats = np.array(nbeats)
# two measures is more fun
#beats = beats.reshape((-1,beats.shape[1]*2,20))
print('two measures', beats.shape)
x_train, x_test, _, _ = train_test_split(beats, beats, test_size=0.25, random_state=0)
print('size', x_train.shape, x_test.shape)

# encoder
input_dim = 20
inputs = Input(shape=(x_train.shape[1], input_dim))
encoded = Reshape((x_train.shape[1]*x_train.shape[2],))(inputs)
encoded = Dropout(0.05)(encoded)
encoded = Dense(256)(encoded)
encoded = Dense(64)(encoded)
encoded = Dense(8)(encoded)
# decoder
decoded = Dense(64)(encoded)
decoded = Dense(256)(decoded)
decoded = Dense(x_train.shape[1]*x_train.shape[2])(decoded)
decoded = Reshape((x_train.shape[1], 20))(decoded)
m = Model(inputs, decoded)
e = Model(inputs, encoded)
enc_in = Input(shape=(8,))
d = Model(enc_in, m.layers[-1](m.layers[-2](m.layers[-3](m.layers[-4](enc_in)))))
# compile
m.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
m.summary()
# train
m.fit(x_train,
      x_train,
      nb_epoch=200,
      batch_size=256,
      shuffle=True,
      validation_data=(x_test, x_test))

enc = e.predict(beats)
cov = np.cov(enc.T)
med = np.mean(enc, axis=0)

# save model
encoder_arch = e.to_json()
decoder_arch = d.to_json()
open(ROOT+'/models/'+STYLE+'/decoder_arch.json', 'w').write(decoder_arch)
d.save_weights(ROOT+'/models/'+STYLE+'/decoder_weights.h5')
# save distibution
np.savez_compressed(ROOT+'/models/'+STYLE+'/distr.npz', med= med, cov= cov)


