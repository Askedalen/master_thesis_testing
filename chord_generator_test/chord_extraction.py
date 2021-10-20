from config import *
import pretty_midi
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import _pickle as pickle
import load_data as load

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_chord_note_names(chord):
    string_name = ""
    for i in range(len(chord)):
            if chord[i]: string_name += notes[i]
    return string_name

def get_chord_name(chord):
    major = [0, 4, 7]
    minor = [0, 3, 7]
    sus2  = [0, 2, 7]
    sus4  = [0, 5, 7]
    minb5 = [0, 3, 6]

    chord_names = []

    chord_shifted = [[i, np.roll(chord, -i)] for i in range(len(chord)) if chord[i]]
    for c in chord_shifted:
        chord_name = notes[c[0]]
        if   (c[1][major]).all(): chord_name += ""
        elif (c[1][minor]).all(): chord_name += "m"
        else:
            if   (c[1][sus2] ).all(): chord_name += "sus2"
            elif (c[1][sus4] ).all(): chord_name += "sus4"
            elif (c[1][minb5]).all(): chord_name += "mb5"
            else: chord_name = ""

        if chord_name != "":
            if   c[1][10]: chord_name += "7"
            elif c[1][11]: chord_name += "maj7"
        chord_names.append(chord_name)
    names = [n for n in chord_names if n]

    if len(names) == 0:
        string_name = get_chord_note_names(chord)
    else:
        string_name = names[0]
    return string_name

def get_chroma(piano_roll):
    piano_roll[piano_roll > 0] = 1
    chroma = np.zeros((12, piano_roll.shape[1]))
    
    for i in range(12):
        chroma[i, :] = np.sum(piano_roll[i::12], axis=0)
    return chroma

def get_chord(chroma, bar, steps, notes_in_chord=3):
    hist = np.sum(chroma[:, bar*steps:(bar+1)*steps], axis=1)
    most_common_notes = np.argpartition(hist, -notes_in_chord)[-notes_in_chord:]
    chord = np.zeros(12)
    chord[most_common_notes] = 1
    return chord

def plot_most_common_chords(num_songs=0):
    data = load.load_data('combined', num_songs)
    all_chords = []
    for song in data:
        chroma = get_chroma(song)

        chords = []
        #Find chords
        steps = 16
        chords_to_include = 10
        for bar in range(math.floor(chroma.shape[1] / steps)):
            chord = get_chord(chroma, bar, steps)
            chords.append(chord)
            all_chords.append(chord)
        # Extract unique chords
    unique_chords, counts= np.unique(np.array(all_chords), axis=0, return_counts=True)
    most_common_idx = np.argpartition(counts, -chords_to_include)[-chords_to_include:]
    most_common_chords = unique_chords[most_common_idx]
    most_common_counts = counts[most_common_idx]

    most_common_names = []

    for chord in most_common_chords:
        string_name = get_chord_name(chord)
        most_common_names.append(string_name)
    

    plt.figure(figsize=(15, 10))
    plt.bar(most_common_names, most_common_counts)
    plt.savefig("most_common_chords.png")

def create_chord_dict(num_songs=0):
    data = load.load_data('pianoroll', num_songs)
    print('Generating chord dictionary...')
    all_chords = []
    start_time = time.time()
    for song in data:
        chroma = get_chroma(song)

        steps = 16
        chords_to_include = 100-1
        for bar in range(math.floor(chroma.shape[1] / steps)):
            chord = get_chord(chroma, bar, steps)
            chord_name = get_chord_name(chord)
            all_chords.append(chord_name)
    end_time = time.time()
    unique_chords, counts = np.unique(np.array(all_chords), axis=0, return_counts=True)
    most_common_idx = np.argpartition(counts, -chords_to_include)[-chords_to_include:]
    most_common_chords = unique_chords[most_common_idx]
    chord_dict = dict()
    chord_dict['UNK'] = 0
    #chord_dict['SUS'] = 1
    for chord in most_common_chords:
        chord_dict[chord] = len(chord_dict)
    
    pickle.dump(chord_dict, open('chord_dict.pickle', 'wb'))
    print(f'Generated chord dictionary in {end_time - start_time} seconds.')

    return chord_dict

def get_chord_dict():
    if os.path.exists('chord_dict.pickle'):
        chord_dict = pickle.load(open('chord_dict.pickle', 'rb'))
    else:
        chord_dict = create_chord_dict()
    chord_to_index = chord_dict
    index_to_chord = dict((v,k) for k,v in chord_dict.items())
    return chord_to_index, index_to_chord

def create_chord_progressions(num_songs = 0):
    files = load.list_pickle_files('pianoroll', num_songs)
    chord_dict, _ = get_chord_dict()
    for i in range(len(files)):
        file = files[i]
        midi_data = pickle.load(open(file, 'rb'))
        chroma = get_chroma(midi_data)
        
        steps = 16
        chord_progression = np.zeros(chroma.shape[1])
        for bar in range(math.floor(chroma.shape[1] / steps)):
            chord = get_chord(chroma, bar, steps)
            chord_name = get_chord_name(chord)
            if chord_name in chord_dict:
                chord_index = chord_dict[chord_name]
            else:
                chord_index = chord_dict['UNK']
            chord_progression[bar*steps:(bar+1)*steps] = chord_index
            #chord_progression[bar*steps] = chord_index
            #chord_progression[(bar*steps)+1:(bar+1)*steps] = chord_dict['SUS']
        filename = os.path.basename(file)
        path = os.path.join(chord_dir, filename)
        pickle.dump(chord_progression, open(path, 'wb'))

        if i % 100 == 0:
            print("Finished {} songs".format(i))

if __name__ == '__main__':
    #plot_most_common_chords(1000)
    create_chord_dict()
    create_chord_progressions()
    #test = get_chord_dict()
    print()