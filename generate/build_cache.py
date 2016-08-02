from pymongo import MongoClient
import numpy as np
import utils
import os
# load files
ROOT = os.path.dirname(os.path.realpath(__file__))

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

# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset

# generate compressed numpy array for each class of beats
for k, v in clmap.iteritems():
    db_beats = collection.find({'class': k})
    class_seq = []
    for i, b in enumerate(db_beats):
        # get np_seq
        np_seq = utils.decompress(b['zip'], b['bar'])
        class_seq.append(np_seq)
    class_seq = np.array(class_seq)
    np.savez_compressed(ROOT + '/cache/'+v+'.npz', class_seq)
    print('saved', v)