from keras.layers import Input, Dense, GRU, TimeDistributed, Flatten, RepeatVector, Reshape
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

FNULL = open(os.devnull, 'w')
ROOT = os.path.dirname(os.path.realpath(__file__))
# for playback and tests
beats = np.load(ROOT + '/cache/latin.npz')
beats = beats['arr_0']

# transform bars and keep them unique
print('clean beats ...')
nbeats = []
compress = []
bar = progressbar.ProgressBar()
print('mean', beats.mean())
for beat in bar(beats):
    for bar in beat:
    #bar = random.choice(beat)
        cp = utils.compress(bar)
        if cp not in compress and bar.mean() > 0.009:
            nbeats.append(bar)
            compress.append(cp)
beats = np.array(nbeats)

x_train, x_test, _, _ = train_test_split(beats, beats, test_size=0.25, random_state=0)
print('size', x_train.shape, x_test.shape)

# encoder
input_dim = 20
inputs = Input(shape=(128, input_dim))
encoded = Reshape((2560,))(inputs)
encoded = Dense(256)(encoded)
encoded = Dense(64)(encoded)
encoded = Dense(8)(encoded)
# decoder
decoded = Dense(64)(encoded)
decoded = Dense(256)(decoded)
decoded = Dense(2560)(decoded)
decoded = Reshape((128, 20))(decoded)
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
      batch_size=512,
      shuffle=True,
      validation_data=(x_test, x_test))

enc = e.predict(beats)
cov = np.cov(enc.T)
med = np.mean(enc, axis=0)

rnd = np.random.multivariate_normal(med, cov, 10)
news = d.predict(rnd)

news = np.around(news)
for i, bd in enumerate(news):
    print(utils.draw(bd))
    mf = utils.np_seq2mid(bd)
    mf.open(ROOT+'/../out/'+str(i)+'.mid', 'wb')
    mf.write()
    mf.close()


