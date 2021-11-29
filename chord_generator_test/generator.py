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
import config as conf
import _pickle as pickle

np.random.seed(2021)

def chord_data_generator(song_list, batch_size=8, max_steps=8, vocabulary=100, infinite=True, rand_data=False, **args):
    chord_dim = (max_steps, 1)
    while True:
        np.random.shuffle(song_list)
        batch_chord_inputs = []
        batch_melody_inputs = []
        batch_targets = []
        for i, song in enumerate(song_list):
            #if (i+1) % 100 == 0:
            #    print(f'Loaded {i+1} songs.')
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

def poly_data_generator(song_list, chord_embedding=None, batch_size = 8, max_steps=128, chord_interval=16, num_notes=60, infinite=True, rand_data=False, **args):
    while True:
        np.random.shuffle(song_list)
        batch_chords = []
        batch_inputs = []
        batch_targets = []
        for i, song in enumerate(song_list):
            #if (i+1) % 100 == 0:
            #    print(f'Loaded {i+1} songs.')
            instrument_data = load_data.get_instruments(song, binary=True, rand_data=rand_data)
            chord_data, melody_data = load_data.get_chords_and_melody(song, binary=True, rand_data=rand_data)
            num_sequences = math.floor(instrument_data.shape[0] / max_steps)
            
            if chord_embedding is not None:
                embedded_chords = chord_embedding.embed_chords_song(chord_data)
            melody_expanded = np.reshape(melody_data, (-1, num_notes))

            for i in range(num_sequences):
                current_step = math.floor((i*max_steps)/chord_interval)
                next_step = math.floor(((i+1)*max_steps)/chord_interval)
                seq_melody = melody_expanded[:-1][i*max_steps:(i+1)*max_steps,]
                counter = to_categorical(np.tile(range(chord_interval), math.floor(max_steps/chord_interval)), num_classes=chord_interval)
                if chord_embedding is not None:
                    seq_chords = embedded_chords[:-1][current_step:next_step,]
                    seq_chords_expanded = np.repeat(seq_chords, chord_interval, axis=0)
                    X = np.concatenate((seq_chords_expanded, seq_melody, counter), axis=1)
                else:
                    seq_chords = chord_data[:-1][current_step:next_step,]
                    seq_chords_expanded = np.repeat(seq_chords, chord_interval, axis=0)
                    seq_chords_expanded = np.reshape(seq_chords_expanded, (max_steps, 1))
                    batch_chords.append(seq_chords_expanded)
                    X = np.concatenate((seq_melody, counter), axis=1)
                y = instrument_data[1:][i*max_steps:(i+1)*max_steps,]
                batch_inputs.append(X)
                batch_targets.append(y)

                if len(batch_inputs) == batch_size:
                    X_out = np.array(batch_inputs)
                    y_out = np.array(batch_targets)
                    if chord_embedding is not None:
                        yield X_out, y_out
                    else:
                        chord_out = np.array(batch_chords)
                        yield [chord_out, X_out], y_out
                    batch_chords = []
                    batch_inputs = []
                    batch_targets = []
        if not infinite:
            break

def embed_poly_chords(chords, Xs, chord_embedding, chord_interval=16, **args):
    batches = []
    emb_start = time.time()
    embedded_chords = chord_embedding.embed_chords_song(np.array(chords).flatten())
    emb_end = time.time()
    print(f'emb time:{emb_end - emb_start}')
    for step in range(len(Xs)):
        s_start = time.time()
        for batch in range(len(Xs[step])):
            embedded_chords_expanded = np.repeat(embedded_chords, chord_interval, axis=0)
            concat = np.concatenate((Xs[step][batch], embedded_chords_expanded), axis=1)
            batches.append(concat)

    return batches

def count_steps(filenames, batch_size = 8, generator_num = 0, chord_embedding = None, **params):
    previous_counts = []
    if os.path.exists('step_counts.pickle'):
        previous_counts = pickle.load(open('step_counts.pickle', 'rb'))
        for steps in previous_counts:
            if steps['generator'] == generator_num and \
               steps['batch_size'] == batch_size and \
               steps['max_steps'] == params['max_steps'] and \
               steps['chord_interval'] == params['chord_interval'] and \
               steps['num_files'] == len(filenames):
                return steps['num_steps']

    if generator_num == 0:
        generator = chord_data_generator(filenames, batch_size=batch_size, infinite=False, **params)
    elif generator_num == 1:
        generator = poly_data_generator(filenames, chord_embedding, batch_size=batch_size, infinite=False, **params)
    num_batches = 0
    for data in generator:
        num_batches += 1

    previous_counts.append({'generator':generator_num, 
                            'batch_size':batch_size,
                            'max_steps':params['max_steps'],
                            'chord_interval':params['chord_interval'],
                            'num_files':len(filenames), 
                            'num_steps':num_batches})
    pickle.dump(previous_counts, open('step_counts.pickle', 'wb'))

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
        model_path = os.path.join(conf.chord_model_dir, 'epoch001.hdf5')
        chord_embedding = load_data.ChordEmbedding(model_path)
        test_generator = poly_data_generator(filenames, chord_embedding, batch_size=4, infinite=False)
        for input, output in test_generator:
            pass#print(input.shape, output.shape)