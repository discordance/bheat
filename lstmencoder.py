from keras.models import Model
from keras import regularizers
from keras.layers import Input, LSTM, RepeatVector, GRU, Dense, Masking, BatchNormalization
from sklearn.cross_validation import train_test_split
import numpy as np
import os

# load stuff
ROOT = os.path.dirname(os.path.realpath(__file__))
themes = np.load(ROOT + '/cache/themes128.npz')
themes = themes['arr_0']
themes = themes
# remove velocitys, this adds noise
themes = np.delete(themes, np.s_[15::], 2)
#themes = themes/np.amax(themes)

##################
# LSTM AUTOENCODER
##################
# this is the size of our encoded representations
seq_encoding_dim = 32  # 32 floats
seq_inputs = Input(shape=(128, 15))
masking = Masking(mask_value=0.)(seq_inputs)
seq_encoded = GRU(seq_encoding_dim, init='he_normal', activation='relu')(masking)
bm = BatchNormalization()(seq_encoded)
seq_decoded = RepeatVector(128)(bm)
seq_decoded = GRU(15, return_sequences=True, init='he_normal', activation='hard_sigmoid')(seq_decoded)
seq_autoencoder = Model(seq_inputs, seq_decoded)
seq_encoder = Model(seq_inputs, seq_encoded)
seq_autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

x_train, x_test, _, _ = train_test_split(themes, themes, test_size=0.33, random_state=0)
seq_autoencoder.summary()
seq_autoencoder.fit(x_train, x_train,
                    nb_epoch=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

score = seq_autoencoder.evaluate(x_test, x_test, batch_size=256)
print('Test score:', score)
print(themes[3][0])
print('-----')
print(seq_autoencoder.predict(np.array([themes[3]]))[0][0])
# seq_encoded_themes = seq_encoder.predict(themes)
# # # put it in cache
#np.savez_compressed(ROOT +'/cache/lstm32_themes128.npz', seq_encoded_themes)




