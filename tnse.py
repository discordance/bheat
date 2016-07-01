import numpy as np
import os
from bhtsne import tsne

ROOT = os.path.dirname(os.path.realpath(__file__))
# load encoded data
enc_themes = np.load(ROOT + '/cache/gru32_themes128.npz')
enc_themes = enc_themes['arr_0']
print(enc_themes.shape)
# load pure data
# themes = np.load(ROOT + '/cache/themes128.npz')
# themes = themes['arr_0']
# themes = themes.reshape(themes.shape[0], -1,)

Y = tsne(enc_themes, 2, 50.0)

np.savez_compressed(ROOT+'/cache/tsne2_themes128.npz', Y)

