from pymongo import MongoClient
import numpy as np
import utils
import os
# load files
ROOT = os.path.dirname(os.path.realpath(__file__))

# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset
beats = collection.find({'bar': 128, 'class': {'$exists': True}})

for i, b in enumerate(beats):
    # get np_seq
    np_seq = utils.decompress(b['zip'], b['bar'])
    theme = np_seq[b['theme_index']]
    # compute drum mean
    dm = theme.mean(axis=0)
    dm = np.delete(dm, np.s_[15::], 0)
    ntheme = None
    # repare shifted beats
    if b['class'] == 0 and dm[0] > 0.0:
        # potentiatlly shifted beat
        if theme[0][0] > 0.0:
            rotate = 0
            while theme[0][0] == 0.0:
                rotate -= 1
                theme = np.roll(theme, rotate, axis=0)
            # rotated, we fix all the sequence
            np_seq = np.roll(np_seq, rotate, axis=1)
            print(utils.draw(np_seq[b['theme_index']]))
            ntheme = np_seq
    if ntheme is not None:
        collection.update_one( {'_id': b['_id']},
                               {'$set': {'dm': list(dm), 'zip': utils.compress(ntheme)}}
                               )
    else:
        collection.update_one({'_id': b['_id']},
                              {'$set': {'dm': list(dm)}}
                              )
