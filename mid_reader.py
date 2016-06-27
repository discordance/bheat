import os
import operator
from music21 import midi
from music21 import meter
import numpy as np
from scipy.spatial.distance import cdist
from pymongo import MongoClient

# internal package
import utils


# TICK RESOLUTION PER BEAT
BEAT_DIV = 32
# Groups semantics
# 0: KICK DRUM {1}
# 1: SNARE AND RIMS {2}
# 2: TOMS {3}
# 3: WORLD {5}
# 4: HAT/CYMB {4}
# 15 bit array + 5 floats for group velocities
PERC_MAP = {
    # KICK GROUP
    0:  [35, 36],
    # SNARE AND RIMS
    1:  [38, 40],
    2:  [37, 39],
    # TOMS
    3:  [41, 45],  # low
    4:  [47, 48],  # mid
    5:  [43, 50],  # high
    # WORLD
    6:  [61, 64, 66],  # low percs african
    7:  [60, 62, 63, 65],  # high percs african
    8:  [76, 78, 79, 68, 74, 75],  # latin a
    9:  [67, 69, 54, 70, 73, 77, 81],  # latin b
    10: [56, 58, 71, 72],  # unusual, unique
    # HH / CYMB
    11: [42, 44],  # muted hh
    12: [46, 55],  # open hat / splash
    13: [49, 52, 57],  # crash and chinese
    14: [51, 53, 59]  # rides
}
PERC_GROUPS = [
    [0],
    [1, 2],
    [3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14]
]


def get_files(path):
    """
    get the midi files recursively from a path
    """
    files = []
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return [fl for fl in files if fl.endswith(('.mid', '.midi', '.MID'))]


def get_midi(fl):
    """
    Reads a file as a Midifile instance, using music21
    """
    # print('open', fl)
    try:
        m = midi.MidiFile()
        m.open(fl, 'rb')
        m.read()
        m.close()
        return m
    except midi.MidiException:
        return None
    except IndexError:
        return None


def extract_timesigs(m, tqn):
    """
    Analyses time signatures in midi file
    """
    ts_list = []
    time_ct = 0
    if len(m.tracks) == 0:
        return []
    for e in m.tracks[0].events:
        # we have a delta time here
        if e.time is not None:
            abstime = midi.translate.midiToDuration(e.time, tqn)
            time_ct += abstime.quarterLength
        if e.type == 'TIME_SIGNATURE':
            try:
                ts = midi.translate.midiEventsToTimeSignature(e)
                ts_list.append((ts, time_ct))
            except IndexError:
                None
            except meter.MeterException:
                None
    return ts_list


def extract_beats(m):
    """
    Analyses beat tracks in midi file
    """
    btrks = []
    for trk in m.tracks:
        chnls = set(trk.getChannels())
        if {None, 10} == chnls:
            btrks.append(trk)
    return btrks


def get_pitch_stats(btrk):
    """
    Returns the statistics on valid and invalid pitches in drumm tracks
    Will also detect invalid pitch (Invalid as out of range of General Midi Drum)
    """
    invalids = {}
    valids = {}
    # make stats
    for evt in btrk.events:
        if evt.type is 'NOTE_ON':
            # invalid pitch
            if evt.pitch < 35 or evt.pitch > 81:
                if evt.pitch not in invalids:
                    invalids[evt.pitch] = 1
                else:
                    invalids[evt.pitch] += 1
            # valid pitch
            else:
                if evt.pitch not in valids:
                    valids[evt.pitch] = 1
                else:
                    valids[evt.pitch] += 1
    return invalids, valids


def get_percs_span_map(btrk):
    """
    Analyses the percussion span used in this track.
    Create a map to map the percussion to the right numpy index (col dim)
    """
    invalids, valids = get_pitch_stats(btrk)
    # decide from stats
    if len(invalids) >= len(valids) or len(valids) < 1:
        # mostly invalid or empty, discard this track probably not a beat track
        return None
    # sort the valids
    svalids = sorted(valids.items(), key=operator.itemgetter(1))
    svalids.reverse()
    # discard percs that are too much
    if len(svalids) > 12:
        svalids = svalids[:12]

    # extract the percs found in the track, only valid ones
    trk_percs = [tup[0] for tup in svalids]
    # creates the indexmap for this track
    index_map = [None for i in range(15)]
    for index, val in enumerate(index_map):
        for perc in PERC_MAP[index]:
            if perc in trk_percs and index_map[index] is None:
                index_map[index] = perc
                trk_percs.remove(perc)
    # find free indices to assign remaining tracks
    for index, val in enumerate(index_map):
        if index_map[index] is None:
            # find the group
            found = None
            for idx, grp in enumerate(PERC_GROUPS):
                if index in grp:
                    found = idx
            # check if there is percs of this group remaining unaffected
            for perc_index in PERC_GROUPS[found]:
                idx_percs = set(PERC_MAP[perc_index])
                intersect = idx_percs.intersection(set(trk_percs))
                if len(intersect) > 0:
                    index_map[index] = list(intersect)[0]
                    trk_percs.remove(index_map[index])

    return index_map


def numpify_beat(btrk, tqn):
    """
    Transform a midi event list (assuming a drum track) into numpy sparse array
    (*,20)
    first 15 indexes are boolean to specify the precence of a percution on a tick
    the last 5 encodes the medium velocity for each groups
    """
    # create an empty step
    def empty_step():
        return [0 for _ in range(15)] + [0.0 for _ in range(5)]

    # get percussion span
    index_map = get_percs_span_map(btrk)
    if index_map is None:
        return None
    sequence = []
    time_ct = 0.
    for evt in btrk.events:
        if evt.type is 'DeltaTime':
            abstime = midi.translate.midiToDuration(evt.time, tqn)
            time_ct += abstime.quarterLength
        # check note on, and also velocity > 0 because of some midi notations
        elif evt.type is 'NOTE_ON' and evt.velocity > 0:
            seq_index = int(round(time_ct*BEAT_DIV))
            # complete the sequence
            while len(sequence) < seq_index+1:
                sequence.append(empty_step())
            # check the perc index, if none, invalid pitch
            try:
                perc_index = index_map.index(evt.pitch)
                sequence[seq_index][perc_index] = 1
                # get group index to set the velocity avg per group
                group = None
                for idx, grp in enumerate(PERC_GROUPS):
                    if perc_index in grp:
                        group = idx
                group_vel = sequence[seq_index][15+group]
                if group_vel == 0.0:
                    sequence[seq_index][15 + group] = evt.velocity/127.0
                else:
                    sequence[seq_index][15 + group] = (sequence[seq_index][15 + group]+(evt.velocity/127.0))/2
            except ValueError:
                None

    np_seq = np.array(sequence)
    return np_seq


def flatten_beats(seqs):
    """
    Transform many numpy sequences in single sequence
    """
    if len(seqs) is 1:
        return seqs[0]
    lens = []
    for i, seq in enumerate(seqs):
        if seq is not None:
            lens.append((len(seq), i, seq))
    lens.sort(key=lambda tup: tup[0])
    lens.reverse()

    if len(lens) == 0:
        return None

    main_seq = lens[0][2]
    for sub in lens[1:]:
        for i, step in enumerate(sub[2]):
            for j, trig in enumerate(step):
                if main_seq[i][j] <= 0 < trig:
                    main_seq[i][j] = trig
                elif 0 < main_seq[i][j] < 1 and trig > 0:
                    main_seq[i][j] = (trig+main_seq[i][j])/2

    return main_seq


def pad_reshape_trim(seq, bar):
    """
    Pads a numpy beat sequence to match an integral number of bars
    :param seq: Numpy beat sequence
    :param bar: Number of steps in a bar
    :return: padded Numpy beat sequence
    """
    # create an empty step
    def empty_step():
        return [0 for _ in range(15)] + [0.0 for _ in range(5)]
    # pad with zero
    while (float(seq.shape[0])/float(bar)).is_integer() is False:
        seq = np.concatenate((seq, np.array([empty_step()])))
    # reshape to measures
    seq = seq.reshape((-1, bar, 20))
    # trim zero measures
    to_remove = []
    for idx, npbar in enumerate(seq):
        if npbar.mean() == 0:
            to_remove.append(idx)
    seq = np.delete(seq, to_remove, axis=0)
    return seq


def get_stats(seq):
    """
    Compute various statistics about the sequence
    :seq numpy beat sequence, in 3D (bars/steps/trigs):
    :return: dictionnary of stats
    """
    # estimate the diversity of a sample
    def diversity(arr):
        index = {}
        for el in arr:
            index[el] = 0
        return (len(index) - 1) / float(len(arr))

    # calculate a ratio to estimate if onsets are strongely tied to the grid
    def gridicity(arr):
        nb_onsets = 0
        nb_grid = 0
        nb_off_grid = 0
        for x in range(0, len(arr)):
            subarr = arr[x]
            if len(np.where(subarr > 0.0)[0]) > 0:
                nb_onsets += 1
                if x % (32 / 4) == 0:
                    nb_grid += 1
                else:
                    nb_off_grid += 1
        return nb_grid / float(nb_onsets)

    stats = {}
    # time sig (how many steps per bar)
    stats['bar'] = seq.shape[1]
    # we use median and matching distance as it is more robust
    median = np.median(seq, axis=0)
    distances = []
    for idx, npbar in enumerate(seq):
        dist = cdist(median, npbar, 'matching').diagonal().mean()
        distances.append((dist, idx))
    distances.sort(key=lambda tup: tup[0])
    # Diversity (estimation of how many different variations from the median)
    stats['diversity'] = diversity([e[0] for e in distances])
    # Standard Deviation (bars dispertion)
    stats['std'] = np.array([e[0] for e in distances]).std()
    # index of the most probable theme
    stats['theme_index'] = distances[0][1]
    stats['gridicity'] = gridicity(seq.reshape(seq.shape[0]*stats['bar'], 20))
    stats['density'] = seq.mean()
    return stats


# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset


# junk
def junk(fname):
    entry = {
        'fname': fname,
        'junk': True
    }
    collection.insert_one(entry)

skip = 87847
# get the files
fls = get_files('/Users/nunja/Documents/Lab/MIDI/BIGDATAM2')
#shuffle(fls)
# for each file
for findex, f in enumerate(fls[skip:]):
    findex += skip
    # check if processed
    if collection.count({'fname': f}) == 0:
        # parse midi
        mf = get_midi(f)
        if mf:
            # get the beat midi tracks
            beat_tracks = extract_beats(mf)
            # get all time signatures
            time_sigs = extract_timesigs(mf, mf.ticksPerQuarterNote)
            # focus on non too complex timesig setups
            if len(time_sigs) == 1 and len(beat_tracks) > 0:
                # converting midi to numpy
                # sometimes the beat is split on many stracks, we have to flatten that
                np_seqs = []
                for beat_track in beat_tracks:
                    np_seqs.append(numpify_beat(beat_track, mf.ticksPerQuarterNote))
                flat_seq = flatten_beats(np_seqs)
                if flat_seq is not None:
                    # if everything is okay
                    # time sig to numbr of steps per bar
                    ts = time_sigs[0][0]
                    bar_ticks = int(ts.barDuration.quarterLength*BEAT_DIV)
                    # clean a bit the sequence
                    np_seq = pad_reshape_trim(flat_seq, bar_ticks)
                    if np_seq.shape[0] == 0:
                        junk(f)
                        continue
                    # compress it for mongo
                    zp = utils.compress(np_seq)
                    # check for duplicate
                    if collection.count({'zip': zp}) > 0:
                        junk(f)
                        print('found a dup, skipping ...')
                    else:
                        # get heuristics
                        entry = get_stats(np_seq)
                        entry['zip'] = zp
                        entry['fname'] = f
                        collection.insert_one(entry)
                else:
                    # trash
                    junk(f)
            else:
                # trash
                junk(f)
        else:
            # trash
            junk(f)
    else:
        print('jump',findex+1)

    print(findex+1, '/', len(fls))









# mf = utils.np_seq2mid(np_seqs[0])
# mf.open('test.mid', 'wb')
# mf.write()
# mf.close()
# print(np_seqs[0][128])
# exit()



# a = np.array([
#     [0,0,1,0,0.5],
#     [0,0,0,0,0],
#     [0,0,0,0,0],
#     [0,0,1,0,0.5],
# ])
#
# b = np.array([
#     [1,0,0,0,0.2],
#     [1,0,0,0,0.2],
#     [0,0,0,0,0],
#     [0,0,0,0,0],
# ])
#
# print(flatten_beats([a, b]))
#
# exit()