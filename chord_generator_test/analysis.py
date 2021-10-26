from config import *
import pretty_midi
import midi_functions as mf
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import _pickle as pickle
import load_data as load

def plot_most_common_chords(num_songs=0):
    data = load.load_data('midi_mod', num_songs)
    all_chords = []
    for song in data:
        chroma = mf.get_chroma(song)

        chords = []
        #Find chords
        steps = 16
        chords_to_include = 10
        for bar in range(math.floor(chroma.shape[1] / steps)):
            chord = mf.get_chord(chroma, bar, steps)
            chords.append(chord)
            all_chords.append(chord)
        # Extract unique chords
    unique_chords, counts= np.unique(np.array(all_chords), axis=0, return_counts=True)
    most_common_idx = np.argpartition(counts, -chords_to_include)[-chords_to_include:]
    most_common_chords = unique_chords[most_common_idx]
    most_common_counts = counts[most_common_idx]

    most_common_names = []

    for chord in most_common_chords:
        string_name = mf.get_chord_name(chord)
        most_common_names.append(string_name)
    
    plt.figure(figsize=(15, 10))
    plt.bar(most_common_names, most_common_counts)
    plt.savefig("most_common_chords.png")