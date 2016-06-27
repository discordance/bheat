from pymongo import MongoClient
from sklearn.manifold import TSNE
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import os
import matplotlib.pyplot as plt
import hdbscan

# internal package
import utils
ROOT = os.path.dirname(os.path.realpath(__file__))

if os.path.isfile(ROOT+'/cache/themes128.npz') is False:
    # database
    client = MongoClient('localhost', 27017)
    db = client.bheat
    collection = db.origset
    beats = collection.find({'bar': 128})
    themes = []
    # extract the themes
    for i, beat in enumerate(beats):
        t_index = beat['theme_index']
        bar = beat['bar']
        print('theme', i)
        full_beat = utils.decompress(beat['zip'], bar)
        theme = full_beat[t_index]
        themes.append(np.array(theme, copy=True))

    themes = np.array(themes)
    np.savez_compressed(ROOT+'/cache/themes128.npz', themes)
else:
    print('found a cache file')
    themes = np.load(ROOT+'/cache/themes128.npz')
    themes = themes['arr_0']

themes = themes.reshape(themes.shape[0], -1,)
print(len(themes))
# keras autoencoder
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats
# this is our input placeholder
input_theme = Input(shape=(2560,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.activity_l1(10e-5))(input_theme)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(2560, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input=input_theme, output=decoded)
# this model maps an input to its encoded representation
encoder = Model(input=input_theme, output=encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
# compile
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train, x_test = np.split(themes, 2)

autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_themes = encoder.predict(np.concatenate((x_train, x_test), axis=0))
np.savez_compressed(ROOT+'/cache/enc32_themes128.npz', encoded_themes)
# put it in cache

#

# tsne
# tsne_model = TSNE(n_components=2, random_state=0)
# data_pts = tsne_model.fit_transform(encoded_themes[:1000])
# print(data_pts.shape)
# plt.plot(data_pts[:,0], data_pts[:,1], "o")
# plt.show()

# hdbscan
# clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
# clusterer.fit(encoded_themes[:1000])
#
# plot_kwds = {'alpha': 0.5, 's': 20, 'linewidths': 0}
# sns.set_context('poster')
# sns.set_style('white')
# sns.set_color_codes()
# palette = sns.color_palette("Set2", max(clusterer.labels_)+1)
# zipped = zip(clusterer.labels_, clusterer.probabilities_)
# cluster_colors = [sns.desaturate(palette[col], 1)
#                   if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
#                   zipped]
# print(data_pts.shape, len(zipped), len(cluster_colors))
# plt.scatter(data_pts[:, 0], data_pts[:, 1], c=cluster_colors, **plot_kwds)
# plt.show()



