from config import *
import os
import numpy as np
from numpy.lib.npyio import load
import _pickle as pickle
import time
import matplotlib.pyplot as plt
import math
import midi_functions as mf
import parse_midi

def load_data(subpath, num_songs=0, return_filenames=False):
    data_path = os.path.join(data_dir, subpath)
    data_files = [os.path.join(data_path, path) for path in os.listdir(data_path) if '.pickle' in path]  
    its = len(data_files)
    if num_songs > 0:
        its = num_songs
    midi_x = []
    startTime = time.time()
    for i in range(its):
        midi_x.append(pickle.load(open(data_files[i], 'rb'))) 
    endTime = time.time()
    print("Loaded {} songs in {} seconds.".format(its, endTime - startTime))

    if return_filenames: 
        return midi_x, data_files

    return midi_x

def list_pickle_files(subpath, num_songs=0):
    data_path = os.path.join(data_dir, subpath)
    data_files = [os.path.join(data_path, path) for path in os.listdir(data_path) if '.pickle' in path]
    if num_songs == 0:
        return data_files
    else:
        return data_files[:num_songs]

def get_random_melody(file):
    midi_data = pickle.load(open(file, 'rb'))
    
    candidate_instruments = []
    for i in range(len(midi_data.instruments)):
        inst = midi_data.instruments[i]
        inst_type = mf.get_inst_type(inst)
        if inst_type == mf.inst_types['MELODY']:
            candidate_instruments.append(i)
        else:
            piano_roll = parse_midi.get_inst_roll(midi_data, i)
            pr_flat = np.sum(piano_roll, axis=0)
            num_steps_playing = pr_flat[pr_flat > 0].shape[0]
            if num_steps_playing >= pr_flat.shape[0] * 0.5:
                candidate_instruments.append(i)
    
    if len(candidate_instruments) < 1: return False

    rand_inst = candidate_instruments[np.random.randint(0, len(candidate_instruments))]
    return parse_midi.get_inst_roll(midi_data, rand_inst)

def get_trainval_chords(num_songs = 0):
    chord_data = load_data('chords', num_songs)
    train_count = math.floor((len(chord_data) / 100) * 80)
    train_set = chord_data[:train_count]
    test_set = chord_data[train_count:]
    return train_set, test_set
    
def get_trainval_chords_and_melody(num_songs = 0):
    chord_data, filenames = load_data('chords', num_songs, return_filenames=True)
    chord_melody_data = []
    for i in range(len(chord_data)):
        chords = chord_data[i]
        midi_file = filenames[i].replace('chords', 'midi')
        melody = get_random_melody(midi_file)
        if melody is not False:
            chord_melody_data.append([chords, melody]) #np.insert(melody, 0, chords, axis=0)
    train_count = math.floor((len(chord_melody_data) / 100) * 80)
    train_set = chord_melody_data[:train_count]
    test_set = chord_melody_data[train_count:]
    return train_set, test_set

def get_chords_and_melody(filename):
    chord_path = os.path.join(chord_dir, filename)
    midi_path = os.path.join(midi_dir, filename)

    chord_data = pickle.load(open(chord_path, 'rb'))
    melody_data = get_random_melody(midi_path)

    return chord_data, melody_data

def get_trainval_filenames(num_songs=0):
    data_path = os.path.join(data_dir, 'chords')
    data_files = [path for path in os.listdir(data_path) if '.pickle' in path]
    if num_songs > 0:
        data_files = data_files[:num_songs]

    train_count = math.floor((len(data_files)/100)*80)
    train_set = data_files[:train_count]
    val_set = data_files[train_count:]
    return train_set, val_set

if __name__ == "__main__":
    #test = get_trainval_chords_and_melody()
    
    print()