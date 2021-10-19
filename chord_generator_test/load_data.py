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
        melody_file = filenames[i].replace('chords', 'melody')
        melody = pickle.load(open(melody_file, 'rb'))
        if melody is not False:
            chord_melody_data.append([chords, melody]) #np.insert(melody, 0, chords, axis=0)
    train_count = math.floor((len(chord_melody_data) / 100) * 80)
    train_set = chord_melody_data[:train_count]
    test_set = chord_melody_data[train_count:]
    return train_set, test_set

def get_chords_and_melody(filename, binary=False):
    chord_path = os.path.join(chord_dir, filename)
    melody_path = os.path.join(melody_dir, filename)

    chord_data = pickle.load(open(chord_path, 'rb'))
    melody_data = pickle.load(open(melody_path, 'rb'))

    if binary:
        melody_data[melody_data > 0] = 1

    return chord_data, melody_data

def get_trainval_filenames(num_songs=0):
    data_path = os.path.join(data_dir, 'melodies')
    data_files = [path for path in os.listdir(data_path) if '.pickle' in path]
    if num_songs > 0:
        data_files = data_files[:num_songs]

    train_count = math.floor((len(data_files)/100)*80)
    train_set = data_files[:train_count]
    val_set = data_files[train_count:]
    return train_set, val_set

if __name__ == "__main__":
    
    
    print()