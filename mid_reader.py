import os
import operator
from music21 import midi
from music21 import meter
from random import shuffle

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
    index_map = {i: None for i in range(15)}
    for index, val in index_map.iteritems():
        for perc in PERC_MAP[index]:
            if perc in trk_percs:
                index_map[index] = perc
                trk_percs.remove(perc)

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
        elif evt.type is 'NOTE_ON':
            seq_index = int(round(time_ct*32))
            # complete the sequence
            while len(sequence) < seq_index+1:
                sequence.append(empty_step())
            print(index_map, evt.pitch)
            # check the perc index, if none, invalid pitch
            # try:
            #     perc_index = index_map.keys()[index_map.values().index(evt.pitch)]
            #     print(perc_index)
            # except ValueError:
            #     print('invalid', evt.pitch)



fls = get_files('/Users/nunja/Documents/Lab/MIDI/BIGDATAM2')
ts_stats = {}
shuffle(fls)
for f in fls:
    mf = get_midi(f)
    if mf:
        beat_tracks = extract_beats(mf)
        time_sigs = extract_timesigs(mf, mf.ticksPerQuarterNote)
        if len(time_sigs) == 1 and len(beat_tracks) > 0:
            print "BEATS"
            for beat_track in beat_tracks:
                numpify_beat(beat_track, mf.ticksPerQuarterNote)
