from numpy import random
from tensorflow.python.keras.utils.np_utils import to_categorical
from config import *
import pretty_midi
import load_data
import midi_functions as mf
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import _pickle as pickle

np.random.seed(2020)

def list_midi_files():
    song_list = []
    if os.path.exists('song_list.pickle'):
        song_list = pickle.load(open("song_list.txt", "rb"))
    else:
        path = "lmd_matched"
        for r, d, f in os.walk(path):
            for file in f:
                if '.mid' in file:
                    song_list.append(os.path.join(r, file))
        pickle.dump(song_list, open("song_list.txt", "wb"))
    return song_list


def midi_to_pickle():
    files = list_midi_files()
    num_failed = 0
    
    start_time = time.time()
    for i in range(len(files)):
        file = files[i]
        filename = os.path.basename(file).replace('.mid', '.pickle')
        midi_file = os.path.join(midi_unmod_dir, filename)
        if os.path.exists(midi_file):
            num_failed += 1
            continue

        try:
            midi_data = pretty_midi.PrettyMIDI(file)
        except:
            print(f'Could not load file {file}')
            continue

        # Store PrettyMIDI as pickle
        pickle.dump(midi_data, open(midi_file, 'wb'))

        if (i+1) % 100 == 0:
            print(f"Finished {i+1}/{len(files)} songs")

    end_time = time.time()
    print(f"Finished {len(files)} songs in {end_time - start_time} seconds. {num_failed} songs failed.")


def normalize_keys(num_files=0):
    files = load_data.list_pickle_files('midi_unmod', num_files)
    start_time = time.time()
    for i in range(len(files)):
        file = files[i]
        filename = os.path.basename(file)
        midi_mod_file = os.path.join(midi_mod_dir, filename)
        if os.path.exists(midi_mod_file):
            continue

        midi_data = pickle.load(open(file, 'rb'))
        key, scale_name = mf.get_key_and_scale(midi_data)
        
        #Get modulated pianoroll
        if key > 0:
            midi_data_modulated = mf.modulate(midi_data, key)
        else:
            midi_data_modulated = midi_data

        #Check if mode is ionian or aeolian
        if scale_name == 'ionian':
            chroma_mod = mf.get_chroma(midi_data_modulated)
            histogram_mod = np.zeros(12)
            chroma_mod[chroma_mod > 0] = 1
            histogram_mod += np.sum(chroma_mod, axis=1)

            if histogram_mod[9] > histogram_mod[0]:
                midi_data_modulated = mf.modulate(midi_data, key - 3)
                scale_name = 'aeolian'
        
        #Store modulated PrettyMIDI as pickle
        pickle.dump(midi_data_modulated, open(midi_mod_file, 'wb'))
        if (i+1) % 100 == 0:
            print(f"Finished {i+1}/{len(files)} songs")
    end_time = time.time()
    print(f"Finished {len(files)} songs in {end_time - start_time} seconds.")


def create_chord_dict(num_files=0, vocabulary=100, chord_interval=16):
    files = load_data.list_pickle_files('midi_mod', num_files)
    print('Generating chord dictionary...')
    all_chords = []
    start_time = time.time()
    for i in range(len(files)):
        file = files[i]
        song = pickle.load(open(file, 'rb'))
        chroma = mf.get_chroma(song)
        for bar in range(math.floor(chroma.shape[1] / chord_interval)):
            chord = mf.get_chord(chroma, bar, chord_interval)
            chord_name = mf.get_chord_name(chord)
            all_chords.append(chord_name)
        if (i+1) % 100 == 0:
            print(f'Scanned {i+1}/{len(files)} songs.')
    end_time = time.time()
    unique_chords, counts = np.unique(np.array(all_chords), axis=0, return_counts=True)
    most_common_idx = np.argpartition(counts, -(vocabulary-1))[-(vocabulary-1):]
    most_common_chords = unique_chords[most_common_idx]
    chord_dict = dict()
    chord_dict['UNK'] = 0
    for chord in most_common_chords:
        chord_dict[chord] = len(chord_dict)
    
    # Store chord dictionary
    pickle.dump(chord_dict, open('chord_dict.pickle', 'wb'))
    print(f'Generated chord dictionary in {end_time - start_time} seconds.')

def create_ML_data(num_files=0, max_steps=8, chord_interval=16, num_notes=128):
    files = load_data.list_pickle_files('midi_mod', num_files)
    chord_dict, _ = load_data.get_chord_dict()
    num_failed = 0

    start_time = time.time()
    for i in range(len(files)):
        file = files[i]
        filename = os.path.basename(file)
        chords_file = os.path.join(chord_dir, filename)
        melody_file = os.path.join(melody_dir, filename)
        if os.path.exists(melody_file):
            continue

        midi_data_modulated = pickle.load(open(file, 'rb'))

        chords = mf.get_chord_progression(midi_data_modulated, chord_dict)
        melody = mf.get_random_melody(midi_data_modulated)
        if melody is False:
            num_failed += 1
            continue
        #melody = melody.T
        # Add padding for end of melody so length is divisible by chord_interval
        melody_rest_steps = chord_interval - melody.shape[0] % chord_interval
        if melody_rest_steps > 0:
            melody = np.resize(melody.T, (chords.shape[0], chord_interval, num_notes))
            melody[-1, -melody_rest_steps:,] = 0
        else:
            melody = np.reshape(melody.T, (-1, chord_interval, num_notes))

        #Add padding for end of song so length is divisible by max_steps
        chords_rest_steps = (max_steps - (chords.shape[0] % max_steps)) + 1
        if chords_rest_steps > 0:
            chords = np.resize(chords, (chords.shape[0] + chords_rest_steps))
            melody = np.resize(melody, (chords.shape[0], chord_interval, num_notes))
            chords[-chords_rest_steps:] = 0
            melody[-chords_rest_steps:,:] = 0

        # Store chord progression and melody
        pickle.dump(chords, open(chords_file, 'wb'))
        pickle.dump(melody, open(melody_file, 'wb'))

        if (i+1) % 100 == 0:
            print(f"Finished {i+1}/{len(files)} songs")
        
    end_time = time.time()
    print(f"Finished {len(files)} songs in {end_time - start_time} seconds. {num_failed} songs failed.")

def create_random_data(num_songs):
    start_time = time.time()
    print('Generating random chord sequence...')
    random_chord_sequence  = np.random.randint(0,  99, (num_songs, 65))
    print('Generating random melody...')
    random_melody_sequence = np.random.randint(0, 127, (num_songs, 65, 16, 1))
    random_melody_sequence = to_categorical(random_melody_sequence, num_classes=128)

    print('Writing to files...')
    for i in range(num_songs):
        chords_path = os.path.join(random_data_dir, 'chords', f'{i:05d}.pickle')
        melody_path = os.path.join(random_data_dir, 'melodies', f'{i:05d}.pickle')

        pickle.dump(random_chord_sequence[i], open(chords_path, 'wb'))
        pickle.dump(random_melody_sequence[i], open(melody_path, 'wb'))
    end_time = time.time()
    print(f'Generated {num_songs} random songs in {end_time - start_time} seconds.')

if __name__ == "__main__":
    #create_random_data(1000)
    #midi_to_pickle()
    #normalize_keys()
    #create_chord_dict()
    create_ML_data()
    #list_midi_files()
    #print()