from keras.layers import Input, Dense, GRU, TimeDistributed, Flatten, RepeatVector
from keras.models import Model
import numpy as np
import os
from sklearn.cross_validation import train_test_split
# local package
import utils
import progressbar

FNULL = open(os.devnull, 'w')
ROOT = os.path.dirname(os.path.realpath(__file__))
# for playback and tests
beats = np.load(ROOT + '/cache/latin.npz')
beats = beats['arr_0']

# transform bars and keep them unique
nbeats = []
compress = []
bar = progressbar.ProgressBar()
for beat in bar(beats):
    for bar in beat:
        cp = utils.compress(bar)
        if cp not in compress and bar.mean() > 0.009:
            nbeats.append(bar)
            compress.append(cp)
beats = np.array(nbeats)

x_train, x_test, _, _ = train_test_split(beats, beats, test_size=0.12, random_state=0)
print('size', x_train.shape, x_test.shape)

# encoder
input_dim = 20
inputs = Input(shape=(128, 20))
encoded = GRU(input_dim, activation='relu', return_sequences=True)(inputs)
encoded = TimeDistributed(Dense(2, activation='relu'))(encoded)
encoded = Flatten()(encoded)
encoded = Dense(2, activation='relu')(encoded)
# decoder
dec_l1 = Dense(20, activation='relu')
dec_l2 = RepeatVector(128)
dec_l3 = GRU(input_dim, return_sequences=True, activation='hard_sigmoid')
decoded = dec_l3(dec_l2(dec_l1(encoded)))
m = Model(inputs, decoded)
e = Model(inputs, encoded)

enc_in = Input(shape=(2,))
decoder_layer = m.layers[-3]
d = Model(enc_in, m.layers[-1](m.layers[-2](m.layers[-3](enc_in))))

# compile
m.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
m.summary()
# train
m.fit(x_train,
      x_train,
      nb_epoch=5,
      batch_size=256,
      shuffle=True,
      validation_data=(x_test, x_test))

print('Test enc dec')
print(utils.draw(x_train[0]))
encdec = m.predict(np.array([x_train[0]]))
print(x_train[0])
print(encdec[0])
#encdec = np.around(encdec)
#print(utils.draw(encdec[0]))
# encform = e.predict(np.array([x_train[0]]))
# print(encform)
# decform = d.predict(encform)
# decform = np.around(decform)
# print(decform.shape)
# print(utils.draw(decform[0]))




# save
encoder_arch = e.to_json()
decoder_arch = d.to_json()
open(ROOT+'/models/latin_encoder_arch.json', 'w').write(encoder_arch)
open(ROOT+'/models/latin_decoder_arch.json', 'w').write(decoder_arch)
e.save_weights(ROOT+'/models/latin_encoder_W.h5')
d.save_weights(ROOT+'/models/latin_decoder_W.h5')
