import os
import numpy as np
from numpy.lib.npyio import load
import _pickle as pickle
import time
import matplotlib.pyplot as plt
from pyo_testing.MusicGenerator import MusicGenerator
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import reshape, function
import math
import config as conf

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
    def __init__(self, model):
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        self.model.reset_states()
        self.embed_layer_output = function([self.model.layers[2].input], [self.model.layers[2].output])

    def embed_chord(self, chord):
        return self.embed_layer_output([[[chord]]])[0]

    def embed_chords_song(self, chords):
        embeded_chords = []
        for chord in chords:
            embeded_chords.append(self.embed_chord(int(chord)))
        return np.array(embeded_chords)

def load_data(subpath, num_songs=0, return_filenames=False):
    data_path = os.path.join(conf.data_dir, subpath)
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
    data_path = os.path.join(conf.data_dir, subpath)
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
        chord_path = os.path.join(conf.random_data_dir, 'chords', filename)
        melody_path = os.path.join(conf.random_data_dir, 'melodies', filename)
    else:
        chord_path = os.path.join(conf.chord_dir, filename)
        melody_path = os.path.join(conf.melody_dir, filename)

    chord_data = pickle.load(open(chord_path, 'rb'))
    melody_data = pickle.load(open(melody_path, 'rb'))

    if binary:
        melody_data[melody_data > 0] = 1
        melody_data = melody_data.astype(bool)

    return chord_data, melody_data

def get_instruments(filename, binary=False):
    instrument_path = os.path.join(conf.instrument_dir, filename)

    instrument_data = pickle.load(open(instrument_path, 'rb'))

    if binary:
        instrument_data[instrument_data > 0] = 1
        instrument_data = instrument_data.astype(bool)

    return instrument_data

def get_trainval_filenames(num_songs=0):
    if os.path.exists('train_filenames.pickle') and os.path.exists('val_filenames.pickle') and os.path.exists('test_filenames.pickle'):
        train_filenames = pickle.load(open('train_filenames.pickle', 'rb'))
        val_filenames = pickle.load(open('val_filenames.pickle', 'rb'))
        return train_filenames, val_filenames
    else:
        data_path = os.path.join(conf.data_dir, 'melodies')
        data_files = [path for path in os.listdir(data_path) if '.pickle' in path]
        np.random.shuffle(data_files)
        if num_songs > 0:
            data_files = data_files[:num_songs]

        trainval_count = math.floor((len(data_files)/100)*80)
        trainval_filenames = data_files[:trainval_count]
        test_filenames = data_files[trainval_count:]
        train_count = math.floor((len(trainval_filenames)/100)*80)
        train_filenames = trainval_filenames[:train_count]
        val_filenames = trainval_filenames[train_count:]

        pickle.dump(train_filenames, open('train_filenames.pickle', 'wb'))
        pickle.dump(val_filenames, open('val_filenames.pickle', 'wb'))
        pickle.dump(test_filenames, open('test_filenames.pickle', 'wb'))

        return train_filenames, val_filenames

def load_test_data(num_songs=0):
    test_filenames = pickle.load('test_filenames.pickle')
    if num_songs > 0:
        test_filenames = test_filenames[:num_songs]
    melodies = []
    targets = []

    for filename in test_filenames:
        instrument_data = get_instruments(filename, binary=True)
        _, melody_data = get_chords_and_melody(filename, binary=True)
        melody_data = np.reshape(melody_data, (-1, conf.num_notes))
        melodies.append(melody_data)
        targets.append(instrument_data)

    return melodies, targets



def load_midi_unmod():
    files = []
    data_files = [os.path.join(conf.midi_unmod_dir, path) for path in os.listdir(conf.midi_unmod_dir) if '.pickle' in path]
    for file in data_files:
        files.append(pickle.load(open(file, 'rb')))
    print('success')

if __name__ == "__main__":
    """  _, test = get_chords_and_melody('0dd4d2b9fbcf96a0fa363a1918255e58.pickle')
    _, test2 = get_chords_and_melody('0dd4d2b9fbcf96a0fa363a1918255e58.pickle', binary=True)
    pickle.dump(test, open('size_test3.pickle', 'wb'))
    pickle.dump(test2, open('size_test4pickle', 'wb')) """
    print()