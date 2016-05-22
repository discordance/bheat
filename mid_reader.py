import os
from music21 import midi
from music21 import meter
from random import shuffle

def get_files(path):
    """
    get the midi files recursively from a path
    """
    files = []
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root,filename))
    return [fl for fl in files if fl.endswith(('.mid','.midi', '.MID'))]


def get_midi(fl):
    """
    Reads a file as a Midifile instance, using music21
    """
    #print('open', fl)
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


def get_percs_span(btrk):
    """
    Analyses if the percussion span used in this track.
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
    # decide from stats
    return len(valids)
    # if len(invalids) >= len(valids) or len(valids) < 1:
    #     # mostly invalid or empty, discard this track probably not a beat track
    #     return None
    # if len(valids) > 9:
    #     return None


def numpify_beat(btrk):
    """
    Transform a midi event list (assuming a drum track) into numpy sparse array
    (*,12)
    """
    # get percussion span
    get_percs_span(btrk)

fls = get_files('/Users/nunja/Documents/Lab/MIDI/BIGDATAM2')
ts_stats = {}
shuffle(fls)
for f in fls:
    mf = get_midi(f)
    if mf:
        beat_tracks = extract_beats(mf)
        time_sigs = extract_timesigs(mf, mf.ticksPerQuarterNote)
        if len(time_sigs) == 1 and len(beat_tracks) > 0:
            for beat_track in beat_tracks:
                print(beat_track.events[1], get_percs_span(beat_track))
                #numpify_beat(beat_track)




