import pretty_midi
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import parse_midi
from config import *

inst_types = {'DRUM':0, 'BASS':1, 'GUITAR':2, 'PIANO':3, 'VOCAL':4, 'MELODY':5, 'OTHER':6}

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

    if searchName(["mel", "melody", "lead"],        [inst_name, track_name]): return inst_types['MELODY']
    if searchName(["bass"],                 [inst_name, track_name]): return inst_types['BASS']
    if searchName(["guit"],                 [inst_name, track_name]): return inst_types['GUITAR']
    if searchName(["pian", "key", "organ"], [inst_name, track_name]): return inst_types['PIANO']
    if searchName(["voc", "voi", "choir"],  [inst_name, track_name]): return inst_types['VOCAL']

    return inst_types['OTHER']

def get_chroma(midi_data):
    initial_tempo = midi_data.get_tempo_changes()[1][0]
    beat_length_seconds = 60 / initial_tempo # Get length of each beat in seconds
    sampling_freq = 1 / (beat_length_seconds / subdivision) # Calculate sampling frequency to get 32nd notes

    piano_roll = midi_data.get_chroma(sampling_freq)
    return piano_roll

def get_piano_roll(midi_data):
    initial_tempo = midi_data.get_tempo_changes()[1][0]
    beat_length_seconds = 60 / initial_tempo # Get length of each beat in seconds
    sampling_freq = 1 / (beat_length_seconds / subdivision) # Calculate sampling frequency to get 32nd notes

    piano_roll = midi_data.get_piano_roll(sampling_freq)
    return piano_roll

def get_all_inst_rolls(midi_data):
    initial_tempo = midi_data.get_tempo_changes()[1][0]
    beat_length_seconds = 60 / initial_tempo # Get length of each beat in seconds
    sampling_freq = 1 / (beat_length_seconds / subdivision) # Calculate sampling frequency to get 32nd notes
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
    sampling_freq = 1 / (beat_length_seconds / subdivision) # Calculate sampling frequency to get 32nd notes
    piano_roll_shape = midi_data.get_piano_roll(sampling_freq).shape

    inst = midi_data.instruments[inst_num]
    pianoroll = np.zeros(piano_roll_shape)
    inst_roll = inst.get_piano_roll(sampling_freq)
    pianoroll[:, :inst_roll.shape[1]] += inst_roll
    return pianoroll

def modulate(midi_data, num_steps):
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch -= num_steps
    return midi_data

def get_key_and_scale(midi_data):
    chroma = get_chroma(midi_data)
    #Create histogram
    histogram = np.zeros(12)
    chroma[chroma > 0] = 1
    histogram += np.sum(chroma, axis=1)

    dia = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    har = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    mel = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
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
    mel_match = [i for i in range(12) if (scale==mel_scales[i]).all()]

    if len(dia_match) > 0:
        scale_name = "ionian"
        key = dia_match[0]
    elif len(har_match) > 0:
        scale_name = "harmonic minor"
        key = har_match[0]
    elif len(mel_match) > 0:
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

def piano_roll_to_midi(piano_roll):
    fs = 8
    program = 0
    is_drum = False
    num_notes = piano_roll.shape[0]
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)

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
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0

    for note in instrument.notes:
        if note.velocity > 127:
            note.velocity = 127
    midi.instruments.append(instrument)
    return midi

def get_random_melody(midi_data, return_midi = False):
    candidate_instruments = []
    cands_incl_polyphonic = []
    for i in range(len(midi_data.instruments)):
        inst = midi_data.instruments[i]
        inst_type = get_inst_type(inst)
        if inst_type == inst_types['MELODY']:
            candidate_instruments.append(i)
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

if __name__ == "__main__":
    #get_inst_type(None)
    piano_roll = pickle.load(open('chord_generator_test\data\pianoroll\d8e6e749b5aa26eef98b5c668203131f.pickle', 'rb'))
    midi = piano_roll_to_midi(piano_roll)
    if os.path.exists('chord_gen_test.mid'):
        os.remove('chord_gen_test.mid')
    midi.write('chord_gen_test.mid')
    print()