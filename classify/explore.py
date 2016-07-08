import utils
import subprocess
from keras.models import Sequential
from keras.layers import GRU, TimeDistributedDense, Flatten, Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from pymongo import MongoClient
import numpy as np
import operator

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
# get class 0 themes
c0_b = list(collection.find({'bar': 128, 'class': 0}, {'_id': 1, 'theme_index': 1, 'zip': 1, 'bar': 1}))
ids = [t['_id'] for t in c0_b]

c0_themes = []

for t in c0_b:
    # get np_seq
    np_seq = utils.decompress(t['zip'], t['bar'])
    tt = np.copy(np_seq[t['theme_index']])
    c0_themes.append(tt)

c0_themes = np.array(c0_themes)

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
    1: 'mini',
    2: 'punk',
    3: 'latin',
    4: 'rock',
    5: 'techno',
    6: 'jazz',
    7: 'electro'
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
# binarize for GRU
lb = LabelBinarizer()
lb.fit(set_y)
set_y = lb.transform(set_y)


t_themes = np.delete(themes, np.s_[15::], 2)
c0_themes = np.delete(c0_themes, np.s_[15::], 2)
# make CV sets
x_train, x_test, y_train, y_test = train_test_split(set_x, set_y, test_size=0.25, random_state=0)
# create sample weights from y_train
nonbin_y_train = lb.inverse_transform(y_train)
count = dict(zip(*np.unique(nonbin_y_train, return_counts=True)))
tt = nonbin_y_train.shape[0]
samples_weights = []
for lbl in nonbin_y_train:
    samples_weights.append(count[lbl]/float(tt))
samples_weights = 1 - np.array(samples_weights)
# keras GRU
m = Sequential()
m.add(GRU(15, input_shape=(128,15), activation='relu', return_sequences=True))
m.add(TimeDistributedDense(15, activation='relu'))
m.add(Flatten())
m.add(Dense(8, activation='hard_sigmoid'))
m.compile(optimizer='adadelta', loss='binary_crossentropy')
m.summary()
#
# GRU Learning
m.fit(x_train, y_train, sample_weight=samples_weights, nb_epoch=100, batch_size=128, shuffle=True, validation_data=(x_test, y_test))
#
gru_predictions = m.predict(c0_themes)
gru_predictions = lb.inverse_transform(np.around(gru_predictions))

# prepare for scikit
t_themes = t_themes.reshape((-1, t_themes.shape[1]*t_themes.shape[2],))
c0_themes = c0_themes.reshape((-1, c0_themes.shape[1]*c0_themes.shape[2],))
x_train = x_train.reshape((-1, x_train.shape[1]*x_train.shape[2],))
x_test = x_test.reshape((-1, x_test.shape[1]*x_test.shape[2],))
x_train = normalize(x_train)
x_test = normalize(x_test)
y_train = lb.inverse_transform(y_train)
y_test = lb.inverse_transform(y_test)

# SVC
print('start SVC')
clf = SVC(kernel='linear', verbose=True, class_weight="balanced")
clf.fit(x_train, y_train)
print('SVC score', clf.score(x_test, y_test))
svc_predictions = clf.predict(c0_themes)

# RF
print('start Random Forest')
clf = RandomForestClassifier(verbose=False, warm_start=True, oob_score=True, max_features="sqrt", random_state=0, class_weight="balanced")
clf.fit(x_train, y_train)
print('RF score', clf.score(x_test, y_test))
rf_predictions = clf.predict(c0_themes)

# KNN
print('start KNN')
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
print('KNN score', clf.score(x_test, y_test))
knn_predictions = []
for i, t in enumerate(c0_themes):
    pred = clf.predict([t])
    knn_predictions.append(pred[0])
    print('knn', i)

knn_predictions = np.array(knn_predictions)

print('gru', gru_predictions.shape, 'svc', svc_predictions.shape, 'rf', rf_predictions.shape, 'knn', knn_predictions.shape)
np.savez_compressed(ROOT + '/cache/labels_c0.npz', gru_predictions, svc_predictions, rf_predictions, knn_predictions)

def vote(votes, w):
    sc = {}
    for i, v in enumerate(votes):
        if v not in sc:
            sc[v] = w[i]
        else:
            sc[v] += w[i]
    sorted_x = sorted(sc.items(), key=operator.itemgetter(1))[::-1]
    return sorted_x[0][0]

for i, t in enumerate(c0_themes):
    print(i, 'gru', clmap[gru_predictions[i]], 'svc', clmap[svc_predictions[i]], 'rf', clmap[rf_predictions[i]], 'knn', clmap[knn_predictions[i]])
    # the latin hack, i want my latin king
    if svc_predictions[i] == 3:
        result = vote([gru_predictions[i], svc_predictions[i], rf_predictions[i], knn_predictions[i]],
                      [0.0, 1.0, 0.1, 0.5])
    else:
        result = vote([gru_predictions[i], svc_predictions[i], rf_predictions[i], knn_predictions[i]], [0.7, 1.0, 0.4, 0.7])
    print('vote result', clmap[result])
    collection.update_one({'_id': ids[i]}, {'$set': {'class': result}})
    print(utils.draw(themes[i]))
    # mf = utils.np_seq2mid(themes[i])
    # mf.open('/tmp/tmp.mid', 'wb')
    # mf.write()
    # mf.close()
    # subprocess.call("/usr/local/bin/timidity -D 0 -R 1000 /tmp/tmp.mid", stdout=FNULL, stderr=FNULL, shell=True)

