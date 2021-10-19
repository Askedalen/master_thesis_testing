import os
import time
import math
from typing import Sequence
import numpy as np
from numpy.lib import utils
import pretty_midi
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.module.module import Module
import load_data
import tensorflow.keras.utils

np.random.seed(2021)

def chord_data_generator(song_list, batch_size = 8, max_steps=1000, num_notes=128, vocabulary=100):
    chord_dim = (max_steps, 1)
    mel_dim = (max_steps, 1, num_notes)
    target_dim = (max_steps, vocabulary)
    num_batches = math.floor(len(song_list) / batch_size)
    while True:
        np.random.shuffle(song_list)
        for batch in range(num_batches):
            chord_inputs = []
            melody_inputs = []
            targets = []
            batch_songs = song_list[batch*batch_size:((batch+1)*batch_size)]
            for file in batch_songs:
                X1 = np.zeros(chord_dim, dtype=int)
                X2 = np.zeros(mel_dim, dtype=int)
                y = np.zeros(target_dim, dtype=int)

                chord_data, melody_data = load_data.get_chords_and_melody(file, binary=True)
                num_steps = chord_data.shape[0]
                if melody_data is False:
                    melody_data = np.zeros((num_notes,num_steps))
                chord_data = np.array(chord_data)
                if num_steps > max_steps:
                    X1 = np.reshape(chord_data[:-1], (-1, 1))[:max_steps,]
                    X2 = np.reshape(melody_data.T[:-1], [-1, 1, num_notes])[:max_steps,]
                    y = to_categorical(chord_data[1:], num_classes=vocabulary)[:max_steps,]
                else:
                    X1[:num_steps-1] = np.reshape(chord_data[:-1], (-1, 1))
                    X2[:num_steps-1] = np.reshape(melody_data.T[:-1], [-1, 1, num_notes])
                    y[:num_steps-1] = to_categorical(chord_data[1:], num_classes=vocabulary)
                chord_inputs.append(X1)
                melody_inputs.append(X2)
                targets.append(y)

            X1_out = np.array(chord_inputs)
            X2_out = np.array(melody_inputs)
            y_out = np.array(targets)
            yield [X1_out, X2_out], y_out

def poly_data_generator(song_list, batch_size = 8, max_steps=1000, num_notes=128, vocabulary=100):
    chord_dim = (max_steps, 1)
    mel_dim = (max_steps, 1, num_notes)
    target_dim = (max_steps, vocabulary)
    num_batches = math.floor(len(song_list) / batch_size)
    while True:
        np.random.shuffle(song_list)
        for batch in range(num_batches):
            chord_inputs = []
            melody_inputs = []
            targets = []
            batch_songs = song_list[batch*batch_size:((batch+1)*batch_size)]
            for file in batch_songs:
                X1 = np.zeros(chord_dim, dtype=int)
                X2 = np.zeros(mel_dim, dtype=int)
                y = np.zeros(target_dim, dtype=int)

                chord_data, melody_data = load_data.get_chords_and_melody(file, binary=True)
                num_steps = chord_data.shape[0]
                if melody_data is False:
                    melody_data = np.zeros((num_notes,num_steps))
                chord_data = np.array(chord_data)
                if num_steps > max_steps:
                    X1 = np.reshape(chord_data[:-1], (-1, 1))[:max_steps,]
                    X2 = np.reshape(melody_data.T[:-1], [-1, 1, num_notes])[:max_steps,]
                    y = to_categorical(chord_data[1:], num_classes=vocabulary)[:max_steps,]
                else:
                    X1[:num_steps-1] = np.reshape(chord_data[:-1], (-1, 1))
                    X2[:num_steps-1] = np.reshape(melody_data.T[:-1], [-1, 1, num_notes])
                    y[:num_steps-1] = to_categorical(chord_data[1:], num_classes=vocabulary)
                chord_inputs.append(X1)
                melody_inputs.append(X2)
                targets.append(y)

            X1_out = np.array(chord_inputs)
            X2_out = np.array(melody_inputs)
            y_out = np.array(targets)
            yield [X1_out, X2_out], y_out
