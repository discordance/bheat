from keras.layers import Input, Dense
from keras.models import Model
from sklearn.cross_validation import train_test_split
import numpy as np
import os
ROOT = os.path.dirname(os.path.realpath(__file__))

themes = np.load(ROOT + '/cache/themes128.npz')
themes = themes['arr_0']
# remove velocitys, this adds noise
themes = np.delete(themes, np.s_[15::], 2)
themes = themes.reshape(themes.shape[0]*themes.shape[1], 15)


# add weights to features,
# Hypothesis: some percussions are more important that others for classification
# weights = np.array([
#     1,    # kik
#     0.9,  # sn
#     0.1,  # rim
#     0.4,  # ltom
#     0.4,  # mtom
#     0.4,  # htom
#     0.2,  # wpercs
#     0.2,  # wpercs
#     0.2,  # wpercs
#     0.2,  # wpercs
#     0.2,  # wpercs
#     0.5,  # chh
#     0.85,  # ohh/splash
#     0.1,  # china
#     0.2   # ride
# ])
# themes = themes*weights
# print(themes.shape, weights.shape)

#x_train, x_test = np.split(themes, 2)
x_train, x_test, _, _ = train_test_split(themes, themes, test_size=0.33, random_state=0)

encoding_dim = 6  # 4 floats
inp = Input(shape=(x_train.shape[1],))
encoded = Dense(encoding_dim, activation='relu')(inp)
decoded = Dense(x_train.shape[1], activation='sigmoid')(encoded)
autoencoder = Model(input=inp, output=decoded)
encoder = Model(input=inp, output=encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
# compile
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.fit(x_train, x_train,
                nb_epoch=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test)
                )
score = autoencoder.evaluate(x_test, x_test, batch_size=128)
print('Test score:', score)
encoded_themes = encoder.predict(themes)
encoded_themes = encoded_themes.reshape((encoded_themes.shape[0]/128, 128, encoding_dim))
# apply weights to beat indexes,
# coefs = []
# for i in range(0, 128):
#     if i%32 == 0:
#         coefs.append(1.0)
#     else:
#         coefs.append(0.25)
# coefs = np.array(coefs)
# for i in range(0, coefs.shape[0]):
#     encoded_themes[:,i,:]=encoded_themes[:,i,:]*coefs[i]
# save
np.savez_compressed(ROOT + '/cache/perc4_themes128.npz', encoded_themes)

