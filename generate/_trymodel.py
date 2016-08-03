from keras.models import model_from_json
import numpy as np
import os
import time
import utils
import subprocess
# local package
np.set_printoptions(threshold=np.inf)

FNULL = open(os.devnull, 'w')
ROOT = os.path.dirname(os.path.realpath(__file__))
STYLE = 'jazz'

print('load decoder ...')
decoder = model_from_json(open(ROOT+'/models/'+STYLE+'/decoder_arch.json').read())
decoder.load_weights(ROOT+'/models/'+STYLE+'/decoder_weights.h5')
print('')
# load beats
print('load dist ...')
distr = np.load(ROOT + '/models/'+STYLE+'/distr.npz')
med = distr['med']
cov = distr['cov']*1.2



rnd = np.random.multivariate_normal(med, cov, 1)
noised = rnd + np.random.normal(0,0.2,8)
start = time.time()
news = np.absolute(decoder.predict(rnd))
variats = np.absolute(decoder.predict(noised))
end = time.time()
print('took', end - start)


new = utils.clean_ml_out(news[0])
variat = utils.clean_ml_out(variats[0])
print(utils.draw(new))
print(utils.draw(variat))
merge = np.array([new,variat]).reshape((1, 256, 20))
mf = utils.np_seq2mid(merge[0])
mf.open('/tmp/tmp.mid', 'wb')
mf.write()
mf.close()
#subprocess.call("/usr/local/bin/timidity -D 0 -R 1000 /tmp/tmp.mid", stdout=FNULL, stderr=FNULL, shell=True)

