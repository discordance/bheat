import utils
import subprocess
from keras.models import Sequential
from keras.layers import GRU, TimeDistributedDense, Flatten, Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from pymongo import MongoClient
import numpy as np

import os
FNULL = open(os.devnull, 'w')
# load files
ROOT = os.path.dirname(os.path.realpath(__file__))
# for playback and tests
themes = np.load(ROOT + '/cache/themes128.npz')
themes = themes['arr_0']
# for labels
labels = np.load(ROOT + '/cache/hdbscan_labels.npz')
labels = labels['arr_0']

# get ids for mongo
# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset
ids = list(collection.find({'bar': 128}, {'_id': 1}))

print('loaded', themes.shape, 'and', labels.shape, 'labels', len(ids), 'ids')

identified = {
    'unknown':  [20, 21, 1, 3, 19, 22],
    'mini':  [5, 8, 31, 30, 33, 36],
    'punk':  [0, 2],
    'latin': [4],
    'rock': [6, 7, 10, 11, 12, 14, 15, 16, 17, 29, 39, 23, 32, 37, 38, 13],
    'techno': [9, 40, 41, 42, 43, 44],
    'jazz': [24, 25, 26, 27, 28, 18, 34, 35],
    'electro': [45, 46]
}

clmap = {
    0: 'unknown',
    1: 'trash',
    2: 'mini',
    3: 'punk',
    4: 'latin',
    5: 'rock',
    6: 'techno',
    7: 'jazz',
    8: 'electro'
}
inv_clmap = {v: k for k, v in clmap.items()}

set_x = []
set_y = []
noise = []
for i, t in enumerate(themes):
    if labels[i] >= 0:
        lab = labels[i]
        for k, v in identified.items():
            if lab in v:
                lab = k
        set_x.append(t)
        set_y.append(inv_clmap[lab])
    else:
        noise.append(i)

# shape x
set_x = np.array(set_x)
set_x = np.delete(set_x, np.s_[15::], 2)

# shape y
set_y = np.array(set_y)
lb = LabelBinarizer()
lb.fit(set_y)
set_y = lb.transform(set_y)
# make CV
x_train, x_test, y_train, y_test = train_test_split(set_x, set_y, test_size=0.25, random_state=0)

# keras
m = Sequential()
m.add(GRU(15, input_shape=(128,15), activation='relu', return_sequences=True))
m.add(TimeDistributedDense(15, activation='relu'))
m.add(Flatten())
m.add(Dense(8, activation='hard_sigmoid'))
m.compile(optimizer='adadelta', loss='binary_crossentropy')
m.summary()

# GRU Learning
m.fit(x_train, y_train, nb_epoch=50, batch_size=128, shuffle=True, validation_data=(x_test, y_test))

t_themes = np.delete(themes, np.s_[15::], 2)
predictions = m.predict(t_themes)
predictions = lb.inverse_transform(np.around(predictions))

for i, pred in enumerate(predictions):
    print(i, pred, '< class')
    mf = utils.np_seq2mid(themes[i])
    mf.open('/tmp/tmp.mid', 'wb')
    mf.write()
    mf.close()
    subprocess.call("/usr/local/bin/timidity -D 0 -R 1000 /tmp/tmp.mid", stdout=FNULL, stderr=FNULL, shell=True)


# x_train = normalize(x_train)
# x_test = normalize(x_test)
# clf = SVC(kernel='linear', verbose=True)
# clf.fit(x_train, y_train)
# print('score', clf.score(x_test, y_test))
#
# for nz in noise:
#     theme = themes[nz]
#     theme = theme.reshape((theme.shape[1]*theme.shape[0],))
#     print(nz, clf.predict([theme]))
#     mf = utils.np_seq2mid(themes[nz])
#     mf.open('/tmp/tmp.mid', 'wb')
#     mf.write()
#     mf.close()
#     subprocess.call("/usr/local/bin/timidity -D 0 -R 1000 /tmp/tmp.mid", stdout=FNULL, stderr=FNULL, shell=True)

#
# cluster = 20
# main_idxs = []
# for i, label in enumerate(labels):
#     if label == cluster:
#         main_idxs.append(i)
# seq = themes[main_idxs[0]]
# for inclust in main_idxs[1:]:
#     seq = np.concatenate((seq, themes[inclust]), axis=0)
# print(seq.shape, main_idxs)
# mf = utils.np_seq2mid(seq)
# mf.open('/tmp/tmp.mid', 'wb')
# mf.write()
# mf.close()
# subprocess.call("/usr/local/bin/timidity -D 0 -R 1000 /tmp/tmp.mid", stdout=FNULL, stderr=FNULL, shell=True)
