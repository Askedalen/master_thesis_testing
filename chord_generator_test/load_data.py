from config import *
import os
import numpy as np
from numpy.lib.npyio import load
import _pickle as pickle
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import reshape, function
import math

import midi_functions as mf

def get_chord_dict():
    if os.path.exists('chord_dict.pickle'):
        chord_dict = pickle.load(open('chord_dict.pickle', 'rb'))
    else:
        return None
    chord_to_index = chord_dict
    index_to_chord = dict((v,k) for k,v in chord_dict.items())
    return chord_to_index, index_to_chord

class ChordEmbedding:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.model.reset_states()
        self.embed_layer_output = function([self.model.layers[2].input], [self.model.layers[2].output])
        self.chord_to_index, self.index_to_chords = get_chord_dict()

    def embed_chord(self, chord):
        return self.embed_layer_output([[[chord]]])[0]

    def embed_chords_song(self, chords):
        embeded_chords = []
        for chord in chords:
            embeded_chords.append(self.embed_chord(int(chord)))
        return np.array(embeded_chords)

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

def get_chords_and_melody(filename, binary=False, rand_data=False):
    if rand_data:
        chord_path = os.path.join(random_data_dir, 'chords', filename)
        melody_path = os.path.join(random_data_dir, 'melodies', filename)
    else:
        chord_path = os.path.join(chord_dir, filename)
        melody_path = os.path.join(melody_dir, filename)

    chord_data = pickle.load(open(chord_path, 'rb'))
    melody_data = pickle.load(open(melody_path, 'rb'))

    if binary:
        melody_data[melody_data > 0] = 1

    return chord_data, melody_data

def get_instruments(filename, binary=False, rand_data=False):
    if rand_data:
        instrument_path = os.path.join(random_data_dir, 'chords', filename)
    else:
        instrument_path = os.path.join(instrument_dir, filename)

    instrument_data = pickle.load(open(instrument_path, 'rb'))

    if binary:
        instrument_data[instrument_data > 0] = 1

    return instrument_data

def get_trainval_filenames(num_songs=0, rand_data=False):
    if rand_data:
        data_path = os.path.join(random_data_dir, 'melodies')
    else:
        data_path = os.path.join(data_dir, 'melodies')
    data_files = [path for path in os.listdir(data_path) if '.pickle' in path]
    if num_songs > 0:
        data_files = data_files[:num_songs]

    train_count = math.floor((len(data_files)/100)*80)
    train_set = data_files[:train_count]
    val_set = data_files[train_count:]
    return train_set, val_set

def load_midi_unmod():
    files = []
    data_files = [os.path.join(midi_unmod_dir, path) for path in os.listdir(midi_unmod_dir) if '.pickle' in path]
    for file in data_files:
        files.append(pickle.load(open(file, 'rb')))
    print('success')

if __name__ == "__main__":
    get_instruments('0dd4d2b9fbcf96a0fa363a1918255e58.pickle')
    print()