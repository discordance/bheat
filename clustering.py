import numpy as np
import os
import seaborn as sns
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import AgglomerativeClustering

import utils
import subprocess
FNULL = open(os.devnull, 'w')

# load files
ROOT = os.path.dirname(os.path.realpath(__file__))
# for clustering and viz
tnse_themes = np.load(ROOT + '/cache/tsne2_themes128.npz')
tnse_themes = tnse_themes['arr_0']
# for playback and tests
themes = np.load(ROOT + '/cache/themes128.npz')
themes = themes['arr_0']
#
enc_themes = np.load(ROOT + '/cache/gru32_themes128.npz')
enc_themes = enc_themes['arr_0']

print('loaded', tnse_themes.shape, 'and', themes.shape, 'themes')

plot_kwds = {'alpha': 0.5, 's': 2, 'linewidths': 0}
sns.set_context('poster')
sns.set_style('dark')
sns.set_color_codes()

# hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=55, min_samples=120)
clusterer.fit(enc_themes)

# what is the most populated cluster ?
lab_ct = {}

for label in clusterer.labels_:
    if label not in lab_ct:
        lab_ct[label] = 0
    else:
        lab_ct[label] += 1
lab_ct = lab_ct.items()
lab_ct.sort(key=itemgetter(1))
print(len(lab_ct))
main_label = lab_ct[::-1][5][0]

main_idxs = []
for i, label in enumerate(clusterer.labels_):
    if label == main_label:
        main_idxs.append(i)

seq = themes[main_idxs[0]]
for inclust in main_idxs[1:]:
    seq = np.concatenate((seq, themes[inclust]), axis=0)
print(seq.shape, main_idxs)
mf = utils.np_seq2mid(seq)
mf.open('/tmp/tmp.mid', 'wb')
mf.write()
mf.close()
subprocess.call("/usr/local/bin/timidity -D 0 -R 1000 /tmp/tmp.mid", stdout=FNULL, stderr=FNULL, shell=True)


palette = sns.color_palette("Paired", max(clusterer.labels_)+1)
zipped = zip(clusterer.labels_, clusterer.probabilities_)
cluster_colors = []
# make the colors
for col, sat in zipped:
    if col == main_label:
        cluster_colors.append((0., 0., 0.))
    elif col >= 0:
        cluster_colors.append(sns.desaturate(palette[col], 1))
    else:
        cluster_colors.append((0.5, 0.5, 0.5))

plt.scatter(tnse_themes[:, 0], tnse_themes[:, 1], c=cluster_colors, **plot_kwds)
plt.show()

# agglomerative
# ward = AgglomerativeClustering(n_clusters=20, linkage='ward').fit(tnse_themes)
# palette = sns.color_palette("Set2", max(ward.labels_)+1)
# cluster_colors = [sns.desaturate(palette[col], 1)
#                   if col >= 0 else (0.5, 0.5, 0.5) for col in
#                   ward.labels_]
# plt.scatter(tnse_themes[:, 0], tnse_themes[:, 1], c=cluster_colors, **plot_kwds)
# plt.show()
