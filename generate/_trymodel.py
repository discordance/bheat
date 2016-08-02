from keras.models import Model
from keras.models import model_from_json
import numpy as np
import os
# local package
import utils
import progressbar

FNULL = open(os.devnull, 'w')
ROOT = os.path.dirname(os.path.realpath(__file__))

print('load encoder and decoder ...')
encoder = model_from_json(open(ROOT+'/models/latin_encoder_arch.json').read())
encoder.load_weights(ROOT+'/models/latin_encoder_W.h5')
decoder = model_from_json(open(ROOT+'/models/latin_decoder_arch.json').read())
decoder.load_weights(ROOT+'/models/latin_decoder_W.h5')

# load beats
print('load beats ...')
beats = np.load(ROOT + '/cache/latin.npz')
beats = beats['arr_0']

# transform bars and keep them unique
print('clean beats ...')
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

print('generate encoded data', beats.shape)
encoded = encoder.predict(beats)
mean = encoded.mean(axis=0)
covm = np.cov(encoded.T)
#samples = np.random.multivariate_normal(mean, covm, 20)
samples = encoded[:10]
generated = decoder.predict(samples)

print('clean output', generated.shape)
generated = np.around(generated)
# for i, beat in enumerate(generated):
#     for j, step in enumerate(beat):
#         for k, perc in enumerate(step):
#             if k < 15:
#                 generated[i][j][k] = round(perc)
#             else:
#                 if perc < 0.1:
#                     generated[i][j][k] = 0.0

print('gen midi')
for i, beat in enumerate(generated):
    print(utils.draw(beat))
    # mf = utils.np_seq2mid(beat)
    # mf.open(ROOT+'/../out/'+str(i)+'.mid', 'wb')
    # mf.write()
    # mf.close()




