import os
import time
import math
from typing import Sequence
import numpy as np
from numpy.lib import utils
from numpy.lib.polynomial import poly
import pretty_midi
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.module.module import Module
from load_data import get_trainval_filenames
import load_data
import tensorflow.keras.utils
from config import *

np.random.seed(2021)

def chord_data_generator(song_list, batch_size = 8, max_steps=8, chord_interval=16, num_notes=128, vocabulary=100, infinite=True, rand_data=False):
    chord_dim = (max_steps, 1) # Dimension for Input1
    while True:
        np.random.shuffle(song_list)
        batch_chord_inputs = []
        batch_melody_inputs = []
        batch_targets = []
        for song in song_list:
            current_chords, current_melody = load_data.get_chords_and_melody(song, binary=True, rand_data=rand_data)

            num_sequences = math.floor(current_chords.shape[0] / max_steps)
            
            for i in range(num_sequences):
                current_chords = np.array(current_chords)
                X1 = current_chords[:-1][i*max_steps:(i+1)*max_steps,]
                X1 = np.reshape(X1, (chord_dim))
                X2 = current_melody[:-1][i*max_steps:(i+1)*max_steps,]
                y = to_categorical(current_chords[1:], num_classes=vocabulary)[i*max_steps:(i+1)*max_steps,]
                batch_chord_inputs.append(X1)
                batch_melody_inputs.append(X2)
                batch_targets.append(y)

                if len(batch_chord_inputs) == batch_size:
                    X1_out = np.array(batch_chord_inputs)
                    X2_out = np.array(batch_melody_inputs)
                    y_out = np.array(batch_targets)
                    yield [X1_out, X2_out], y_out
                    
                    batch_chord_inputs = []
                    batch_melody_inputs = []
                    batch_targets = []
        if not infinite:
            break

def poly_data_generator(song_list, chord_embedding, batch_size = 8, max_steps=128, chord_interval=16, num_notes=128, vocabulary=100, infinite=True, rand_data=False):
    while True:
        np.random.shuffle(song_list)
        batch_inputs = []
        batch_targets = []
        for song in song_list:
            instrument_data = load_data.get_instruments(song, binary=True, rand_data=rand_data)
            chord_data, melody_data = load_data.get_chords_and_melody(song, binary=True, rand_data=rand_data)
            num_sequences = math.floor(instrument_data.shape[0] / max_steps)
            
            embedded_chords = chord_embedding.embed_chords_song(chord_data)
            melody_expanded = np.reshape(melody_data, (-1, 128))

            for i in range(num_sequences):
                current_step = math.floor((i*max_steps)/chord_interval)
                next_step = math.floor(((i+1)*max_steps)/chord_interval)
                seq_chords = embedded_chords[:-1][current_step:next_step,]
                seq_chords_expanded = np.repeat(seq_chords, chord_interval, axis=0)
                seq_melody = melody_expanded[:-1][i*max_steps:(i+1)*max_steps,]
                counter = to_categorical(np.tile(range(chord_interval), math.floor(max_steps/chord_interval)), num_classes=chord_interval)
                X = np.concatenate((seq_chords_expanded, seq_melody, counter), axis=1)
                y = instrument_data[1:][i*max_steps:(i+1)*max_steps,]
                batch_inputs.append(X)
                batch_targets.append(y)

                if len(batch_inputs) == batch_size:
                    X_out = np.array(batch_inputs)
                    y_out = np.array(batch_targets)
                    yield X_out, y_out

                    batch_inputs = []
                    batch_targets = []
        if not infinite:
            break

def count_steps(filenames, batch_size = 8, generator = 0, chord_embedding = None, **params):
    if generator == 0:
        generator = chord_data_generator(filenames, batch_size=8, infinite=False, **params)
    elif generator == 1:
        generator = poly_data_generator(filenames, chord_embedding, batch_size=8, infinite=False, **params)
    num_batches = 0
    for data in generator:
        num_batches += 1
    return num_batches

if __name__ == '__main__':
    test_chord = False
    test_poly = True
    filenames, _ = get_trainval_filenames()
    if test_chord:
        test_chord_gen = chord_data_generator(filenames, batch_size=4)
        for [input1, input2], output in test_chord_gen:
            print(input1.shape, input2.shape, output.shape)
    if test_poly:
        model_path = os.path.join(results_dir, 'models', 'epoch085.hdf5')
        chord_embedding = load_data.ChordEmbedding(model_path)
        test_generator = poly_data_generator(filenames, chord_embedding, batch_size=4, infinite=False)
        for input, output in test_generator:
            pass#print(input.shape, output.shape)