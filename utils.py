from music21 import midi
import base64
import numpy as np

BEAT_DIV = 32
TICKS_PER_Q = 192
PERC_MAP = {
    # KICK GROUP
    0:  36,
    # SNARE AND RIMS
    1:  40,
    2:  37,
    # TOMS
    3:  41,  # low
    4:  47,  # mid
    5:  50,  # high
    # WORLD
    6:  64,  # low percs african
    7:  63,  # high percs african
    8:  68,  # latin a
    9:  77,  # latin b
    10: 56,  # unusual, unique
    # HH / CYMB
    11: 42,  # muted hh
    12: 46,  # open hat / splash
    13: 49,  # crash and chinese
    14: 51  # rides
}
PERC_GROUPS = [
    [0],
    [1, 2],
    [3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14]
]



def decompress(str, bar):
    zipped = base64.decodestring(str)
    b64 = zipped.decode('zlib')
    arr = np.frombuffer(base64.decodestring(b64))
    rshaped = arr.reshape(len(arr)/bar/20, bar, 20)
    return rshaped

def compress(seq):
    b64 = base64.b64encode(seq)
    compressed = base64.encodestring(b64.encode('zlib'))
    return compressed

def draw(seq, bar = 64, quarter = 16):
    """
    Draws the seq in the terminal
    :param seq: seq to be drawn
    :return:
    """
    st = ""
    for i in reversed(range(0, 15)):
        for j in range(0, len(seq)):
            if seq[j][i] > 0.0:
                st += 'X'
            else:
                char = ''
                if j % bar == 0:
                    char = '|'
                elif  j % quarter == 0:
                    char = ';'
                else:
                    char = '.'
                st += char
        st += "\n"
    return st


def np_seq2mid(np_seq):
    """
    Converts a numpy array to a midi file.
    :param np_seq: numpy beat sequence
    :return: music21.midi.MidiFile
    """
    mt = midi.MidiTrack(1)
    t = 0
    tlast = 0
    for step in np_seq:
        # onset will be true if at least one trig is > 0.0
        # the remaining trigs are added at the same delta time
        onset = False # we encountered an onset at this step
        for idx, trig in enumerate(step[:15]):
            # find the group
            group = None
            for index, grp in enumerate(PERC_GROUPS):
                if idx in grp:
                    group = index
            if trig > 0.0:
                vel = int(step[15+group]*127)
                pitch = PERC_MAP[idx]
                dt = midi.DeltaTime(mt)
                if onset is False:
                    dt.time = t - tlast
                else:
                    dt.time = 0
                mt.events.append(dt)
                me = midi.MidiEvent(mt)
                me.type = "NOTE_ON"
                me.channel = 10
                me.time = None  # d
                me.pitch = pitch
                me.velocity = vel
                mt.events.append(me)
                if onset is False:
                    tlast = t + 6
                    onset = True
        if onset is True:
            # reset onset for the noteoff
            onset = False
            # makes the note off now
            for idx, trig in enumerate(step[:15]):
                if trig > 0.0:
                    pitch = PERC_MAP[idx]
                    dt = midi.DeltaTime(mt)
                    if onset is False:
                        dt.time = 6
                    else:
                        dt.time = 0
                    mt.events.append(dt)
                    me = midi.MidiEvent(mt)
                    me.type = "NOTE_OFF"
                    me.channel = 10
                    me.time = None  # d
                    me.pitch = pitch
                    me.velocity = 0
                    mt.events.append(me)
                    if onset is False:
                        onset = True
        t += TICKS_PER_Q/BEAT_DIV
    # add end of track
    dt = midi.DeltaTime(mt)
    dt.time = 0
    mt.events.append(dt)
    me = midi.MidiEvent(mt)
    me.type = "END_OF_TRACK"
    me.channel = 1
    me.data = ''  # must set data to empty string
    mt.events.append(me)
    # make midi file
    mf = midi.MidiFile()
    mf.ticksPerQuarterNote = TICKS_PER_Q  # cannot use: 10080
    mf.tracks.append(mt)
    return mf
