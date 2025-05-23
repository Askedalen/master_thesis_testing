import pretty_midi
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import config as conf

inst_types = {'DRUM':0, 'BASS':1, 'GUITAR':2, 'PIANO':3, 'VOCAL':4, 'MELODY':5, 'OTHER':6}
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def searchName(keys, strings):
    for s in strings:
        for k in keys:
            if k.lower() in s.lower():
                return True
    return False

def get_inst_type(inst_data):
    if inst_data.is_drum:
        return inst_types['DRUM']
    inst_name = pretty_midi.program_to_instrument_name(inst_data.program)
    track_name = inst_data.name

    if searchName(["mel", "melody", "lead"], [inst_name, track_name]): return inst_types['MELODY']
    if searchName(["bass", "tuba"],          [inst_name, track_name]): return inst_types['BASS']
    if searchName(["guit"],                  [inst_name, track_name]): return inst_types['GUITAR']
    if searchName(["pian", "key", "organ"],  [inst_name, track_name]): return inst_types['PIANO']
    if searchName(["voc", "voi", "choir"],   [inst_name, track_name]): return inst_types['VOCAL']

    return inst_types['OTHER']

def get_chroma(midi_data):
    initial_tempo = midi_data.get_tempo_changes()[1][0]
    beat_length_seconds = 60 / initial_tempo # Get length of each beat in seconds
    sampling_freq = 1 / (beat_length_seconds / conf.subdivision) # Calculate sampling frequency to get 32nd notes

    try:
        piano_roll = midi_data.get_chroma(sampling_freq)
    except IndexError:
        print('IndexError')
        piano_roll = None
    return piano_roll

def get_piano_roll(midi_data):
    initial_tempo = midi_data.get_tempo_changes()[1][0]
    beat_length_seconds = 60 / initial_tempo # Get length of each beat in seconds
    sampling_freq = 1 / (beat_length_seconds / conf.subdivision) # Calculate sampling frequency to get 32nd notes

    piano_roll = midi_data.get_piano_roll(sampling_freq)
    return piano_roll

def get_all_inst_rolls(midi_data):
    initial_tempo = midi_data.get_tempo_changes()[1][0]
    beat_length_seconds = 60 / initial_tempo # Get length of each beat in seconds
    sampling_freq = 1 / (beat_length_seconds / conf.subdivision) # Calculate sampling frequency to get 32nd notes
    piano_roll_shape = midi_data.get_piano_roll(sampling_freq).shape

    piano_rolls = []

    for inst in midi_data.instruments:
        roll = np.zeros(piano_roll_shape)
        inst_roll = inst.get_piano_roll(sampling_freq)
        roll[:, :inst_roll.shape[1]] += inst_roll
        piano_rolls.append(roll)
    return piano_rolls

def get_inst_roll(midi_data, inst_num):
    initial_tempo = midi_data.get_tempo_changes()[1][0]
    beat_length_seconds = 60 / initial_tempo # Get length of each beat in seconds
    sampling_freq = 1 / (beat_length_seconds / conf.subdivision) # Calculate sampling frequency to get 32nd notes
    piano_roll_shape = midi_data.get_piano_roll(sampling_freq).shape

    inst = midi_data.instruments[inst_num]
    pianoroll = np.zeros(piano_roll_shape)
    if inst.is_drum:
        inst.is_drum = False
        inst_roll = inst.get_piano_roll(sampling_freq)
        inst.is_drum = True
    else:
        inst_roll = inst.get_piano_roll(sampling_freq)
    pianoroll[:, :inst_roll.shape[1]] += inst_roll
    if inst.is_drum:
        pianoroll = pianoroll[36:60,:]
    else:
        pianoroll = pianoroll[conf.pr_start_idx:conf.pr_end_idx,:]
    return pianoroll

def modulate(midi_data, num_steps):
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch -= num_steps
                if note.pitch < 0:
                    note.pitch += 12
                elif note.pitch > 127:
                    note.pitch -= 12
    return midi_data

def get_chord(chroma, bar, steps, notes_in_chord=3):
    hist = np.sum(chroma[:, bar*steps:(bar+1)*steps], axis=1)
    if np.max(hist) == 0:
        return hist
    most_common_notes = np.argpartition(hist, -notes_in_chord)[-notes_in_chord:]
    chord = np.zeros(12)
    chord[most_common_notes] = 1
    return chord

def get_chord_note_names(chord):
    string_name = ""
    for i in range(len(chord)):
            if chord[i]: string_name += notes[i]
    return string_name

def get_chord_name(chord):
    major = [0, 4, 7]
    minor = [0, 3, 7]
    sus2  = [0, 2, 7]
    sus4  = [0, 5, 7]
    minb5 = [0, 3, 6]

    chord_names = []

    if np.max(chord) == 0:
        return 'UNK'

    chord_shifted = [[i, np.roll(chord, -i)] for i in range(len(chord)) if chord[i]]
    for c in chord_shifted:
        chord_name = notes[c[0]]
        if   (c[1][major]).all(): chord_name += ""
        elif (c[1][minor]).all(): chord_name += "m"
        else:
            if   (c[1][sus2] ).all(): chord_name += "sus2"
            elif (c[1][sus4] ).all(): chord_name += "sus4"
            elif (c[1][minb5]).all(): chord_name += "mb5"
            else: chord_name = ""

        if chord_name != "":
            if   c[1][10]: chord_name += "7"
            elif c[1][11]: chord_name += "maj7"
        chord_names.append(chord_name)
    names = [n for n in chord_names if n]

    if len(names) == 0:
        string_name = get_chord_note_names(chord)
    else:
        string_name = names[0]
    return string_name

    

def get_key_and_scale(midi_data):
    chroma = get_chroma(midi_data)
    if chroma is None:
        return None, None
    #Create histogram
    histogram = np.zeros(12)
    chroma[chroma > 0] = 1
    histogram += np.sum(chroma, axis=1)

    dia = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    har = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    mel = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1])
    blu = np.array([1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0])
    dia_scales = []
    har_scales = []
    mel_scales = []
    blu_scales = []
    for i in range(12):
        dia_scales.append(np.roll(dia, i))
        har_scales.append(np.roll(har, i))
        mel_scales.append(np.roll(mel, i))
        blu_scales.append(np.roll(blu, i))

    most_common_notes = np.argpartition(histogram, -7)[-7:]

    scale = np.zeros(12)
    scale[most_common_notes] = 1

    dia_match = [i for i in range(12) if (scale==dia_scales[i]).all()]
    har_match = [i for i in range(12) if (scale==har_scales[i]).all()]

    if len(dia_match) > 0:
        scale_name = "ionian"
        key = dia_match[0]
        if histogram[key - 3] > histogram[key]:
            scale_name = "aeolian"
            key = key - 3
    elif len(har_match) > 0:
        scale_name = "harmonic minor"
        key = har_match[0]
    else:
        most_common_notes = np.argpartition(histogram, -9)[-9:]
        scale = np.zeros(12)
        scale[most_common_notes] = 1
        mel_match = [i for i in range(12) if (scale==mel_scales[i]).all()]
        if len(mel_match) > 0:
            scale_name = "melodic minor"
            key = mel_match[0]
        else:
            most_common_notes = np.argpartition(histogram, -6)[-6:]
            scale = np.zeros(12)
            scale[most_common_notes] = 1
            blu_match = [i for i in range(12) if (scale==blu_scales[i]).all()]
            if len(blu_match) > 0:
                scale_name = "blues"
                key = blu_match[0]
            else:
                scale_name = "undefined"
                key = 0
    return key, scale_name

def piano_roll_to_midi(piano_roll, name, program=0, is_drum=False):
    #This function is taken from https://github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py
    piano_roll = piano_roll.T
    fs = math.floor((conf.tempo / 60) * conf.subdivision)
    num_notes = piano_roll.shape[0]
    instrument = pretty_midi.Instrument(name=name, program=program, is_drum=is_drum)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')
    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(num_notes, dtype=int)
    note_on_time = np.zeros(num_notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note + conf.pr_start_idx,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0

    for note in instrument.notes:
        if note.velocity > 127:
            note.velocity = 127
    return instrument

def get_poly_output_as_midi(model_output):
    pian_pianoroll = model_output[:,:conf.num_notes]
    guit_pianoroll = model_output[:,conf.num_notes:conf.num_notes*2]
    bass_pianoroll = model_output[:,conf.num_notes*2:conf.num_notes*3]
    drum_pianoroll = model_output[:,conf.num_notes*3:]

    pian_midi = piano_roll_to_midi(pian_pianoroll, 'piano', program=5)
    guit_midi = piano_roll_to_midi(guit_pianoroll, 'guitar', program=25)
    bass_midi = piano_roll_to_midi(bass_pianoroll, 'bass', program=34)
    drum_midi = piano_roll_to_midi(drum_pianoroll, 'drums', program=119, is_drum=True)

    midi = pretty_midi.PrettyMIDI()
    midi.instruments.append(pian_midi)
    midi.instruments.append(guit_midi)
    midi.instruments.append(bass_midi)
    midi.instruments.append(drum_midi)

    return midi

def get_random_melody(midi_data, return_midi = False):
    candidate_instruments = []
    cands_incl_polyphonic = []
    for i in range(len(midi_data.instruments)):
        inst = midi_data.instruments[i]
        inst_type = get_inst_type(inst)
        if inst_type == inst_types['MELODY']:
            candidate_instruments.append(i)
        elif inst_type == inst_types['BASS']:
            continue
        else:
            piano_roll = get_inst_roll(midi_data, i)
            piano_roll[piano_roll > 0] = 1
            pr_flat = np.sum(piano_roll, axis=0)
            num_steps_playing = pr_flat[pr_flat > 0].shape[0]
            num_steps_monophonic = pr_flat[pr_flat == 1].shape[0]
            if num_steps_playing >= pr_flat.shape[0] * 0.5 \
            and num_steps_monophonic >= num_steps_playing * 0.9:
                candidate_instruments.append(i)
            elif num_steps_playing >= pr_flat.shape[0] * 0.5:
                cands_incl_polyphonic.append(i)
    
    if len(candidate_instruments) < 1 and len(cands_incl_polyphonic) < 1: 
        return False

    if len(candidate_instruments) < 1:
        rand_inst = cands_incl_polyphonic[np.random.randint(0, len(cands_incl_polyphonic))]
    else:
        rand_inst = candidate_instruments[np.random.randint(0, len(candidate_instruments))]
    if return_midi:
        return get_inst_roll(midi_data, rand_inst), midi_data.instruments[rand_inst]
    else:
        return get_inst_roll(midi_data, rand_inst)

def get_chord_progression(midi_data, chord_dict=None):
    chroma = get_chroma(midi_data)
    chord_interval = 16
    chord_progression = []
    for bar in range(math.floor(chroma.shape[1] / chord_interval)):
        chord = get_chord(chroma, bar, chord_interval)
        chord_name = get_chord_name(chord)
        if chord_dict is not None:
            if chord_name in chord_dict:
                chord_index = chord_dict[chord_name]
            else:
                chord_index = chord_dict['UNK']
            chord_progression.append(chord_index)
        else:
            chord_progression.append(chord_name)
    return np.array(chord_progression)
    
def get_instrument_tracks_combined(midi_data):
    num_steps = get_piano_roll(midi_data).T.shape[0]

    drum_track = np.zeros((num_steps, 24))
    bass_track = np.zeros((num_steps, conf.num_notes))
    guit_track = np.zeros((num_steps, conf.num_notes))
    pian_track = np.zeros((num_steps, conf.num_notes))
    
    for i in range(len(midi_data.instruments)):
        inst_name = get_inst_type(midi_data.instruments[i])
        if inst_name == inst_types['DRUM']:
            drum_roll = get_inst_roll(midi_data, i).T
            drum_track[:drum_roll.shape[0]] += drum_roll
        elif inst_name == inst_types['BASS']:
            bass_roll = get_inst_roll(midi_data, i).T
            bass_track[:bass_roll.shape[0]] += bass_roll
        elif inst_name == inst_types['GUITAR']:
            guit_roll = get_inst_roll(midi_data, i).T
            guit_track[:guit_roll.shape[0]] += guit_roll
        elif inst_name == inst_types['PIANO']:
            pian_roll = get_inst_roll(midi_data, i).T
            pian_track[:pian_roll.shape[0]] += pian_roll
        else:
            continue
    
    comb_track = np.concatenate((pian_track, guit_track, bass_track, drum_track), axis=1)
    
    return comb_track

if __name__ == "__main__":
    #get_inst_type(None)
    midi = pickle.load(open(os.path.join(conf.midi_mod_dir,'04266ac849c1d3814dc03bbf61511b33.pickle'), 'rb'))
    inst = get_instrument_tracks_combined(midi)
    print()