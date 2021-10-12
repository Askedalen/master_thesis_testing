from config import *
import pretty_midi
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import _pickle as pickle


np.random.seed(2020)

subdivision = 4 # Subdivision for each beat

def listFiles():
    files = []
    if os.path.exists('song_list.txt'):
        txt = open("song_list.txt", "r")
        for line in txt:
            files.append(line.replace('\n', ''))
        txt.close()
    else:
        txt = open("song_list.txt", "w")
        path = "lmd_matched"
        for r, d, f in os.walk(path):
            for file in f:
                if '.mid' in file:
                    files.append(os.path.join(r, file))
                    txt.write(os.path.join(r, file))
                    txt.write("\n")
        txt.close()
    return files

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

def midi_to_pickle(num_files=0):
    files = listFiles()
    its = len(files)
    random_start = 0
    if num_files > 0:
        its = num_files
        random_start = np.random.randint(0, len(files) - num_files)

    startTime = time.time()
    for i in range(its):
        file = files[i + random_start]

        try:
            midi_data = pretty_midi.PrettyMIDI(file)
        except:
            print("Could not load file {}".format(file))
            continue

        key, scale_name = get_key_and_scale(midi_data)
        
        #Get modulated pianoroll
        if key > 0:
            midi_data_modulated = modulate(midi_data, key)
        else:
            midi_data_modulated = midi_data

        #Check if mode is ionian or aeolian
        if scale_name == 'ionian':
            chroma_mod = get_chroma(midi_data_modulated)
            histogram_mod = np.zeros(12)
            chroma_mod[chroma_mod > 0] = 1
            histogram_mod += np.sum(chroma_mod, axis=1)

            if histogram_mod[9] > histogram_mod[0]:
                midi_data_modulated = modulate(midi_data, key - 3)
                scale_name = 'aeolian'

        combined_roll = get_piano_roll(midi_data_modulated)
        
        filename = os.path.basename(file).replace('.mid', '.pickle')
        midi_file = os.path.join(midi_dir, filename)
        combined_file = os.path.join(pianoroll_dir, filename)
        pickle.dump(midi_data_modulated, open(midi_file, 'wb'))
        pickle.dump(combined_roll, open(combined_file, 'wb'))

        if i % 10 == 0:
            print("Finished {} songs".format(i))

if __name__ == "__main__":
    midi_to_pickle(1000)
    #parse_midi()