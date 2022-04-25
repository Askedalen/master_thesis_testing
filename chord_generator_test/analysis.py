import config as conf
import pretty_midi
import midi_functions as mf
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import _pickle as pickle
import load_data as load
import time

def plot_most_common_chords(num_songs=0):
    files = load.list_pickle_files('midi_mod', num_songs)
    all_chords = []
    for i in range(len(files)):
        file = files[i]
        song = pickle.load(open(file, 'rb'))
        chroma = mf.get_chroma(song)
        for bar in range(math.floor(chroma.shape[1] / conf.chord_interval)):
            chord = mf.get_chord(chroma, bar, conf.chord_interval)
            chord_name = mf.get_chord_name(chord)
            all_chords.append(chord_name)
    unique_chords, counts = np.unique(np.array(all_chords), axis=0, return_counts=True)
    most_common_idx = np.argpartition(counts, -(10))[-(10):]
    most_common_chords = unique_chords[most_common_idx]
    most_common_counts = counts[most_common_idx]
    print("The most common chords are:")
    for i in range(len(most_common_chords)):
        print(f"{most_common_chords[i]} {most_common_counts[i]}")

    plt.figure()
    plt.bar(range(len(counts)),np.sort(counts)[::-1], width=1.0)
    plt.title('Number of occurrences for each chord')
    plt.xlabel('Chord (sorted)')
    plt.ylabel('Occurrences')
    plt.savefig('chord_occurrences.png')

def plot_most_common_scales(num_songs=0):
    files = load.list_pickle_files('midi_unmod', num_songs)
    counts = {'ionian': 0,
              'aeolian': 0,
              'harmonic minor': 0,
              'melodic minor': 0,
              'blues': 0,
              'undefined': 0}
    scale_names = ['Major', 'Natural\nminor', 'Harmonic\nminor', 'Melodic\nminor', 'Blues', 'Unknown']
    for i in range(len(files)):
        file = files[i]

        midi_data = pickle.load(open(file, 'rb'))
        _, scale_name = mf.get_key_and_scale(midi_data)
        
        try:
            counts[scale_name] += 1
        except:
            print("Error: scale name ", scale_name)
            continue
        if i % 1000 == 0:
            print(f'Recorded the scales of {i} songs')
    plt.figure(figsize=(6.5, 5.5))
    plt.bar(range(len(counts)), list(counts.values()), align='center')
    plt.xticks(range(len(counts)), scale_names)
    plt.title('Number of occurrences for each scale')
    plt.xlabel('Occurrences')
    plt.ylabel('Scale')
    plt.savefig('scale_occurrences.png')

def plot_full_histogram(num_songs=0):
    files = load.list_pickle_files('midi_mod', num_songs)
    hist = np.zeros(128)
    for i in range(len(files)):
        file = files[i]
        song = pickle.load(open(file, 'rb')) 
        pianoroll = mf.get_piano_roll(song)
        pianoroll[pianoroll > 0] = 1
        hist += np.sum(pianoroll, axis=1)
    plt.figure()
    plt.bar(range(128), hist, width=1)
    plt.title('Histogram of all notes')
    plt.xlabel('Note MIDI index')
    plt.ylabel('Nmber of timesteps played')
    plt.savefig('full_hist.png')

def plot_chroma_histogram(num_songs=0):
    files = load.list_pickle_files('midi_mod', num_songs)
    hist = np.zeros(12)
    for i in range(len(files)):
        file = files[i]
        song = pickle.load(open(file, 'rb')) 
        pianoroll = mf.get_chroma(song)
        pianoroll[pianoroll > 0] = 1
        hist += np.sum(pianoroll, axis=1)
    plt.figure()
    plt.bar(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], hist, width=0.9)
    plt.title('Histogram of all unique notes')
    plt.xlabel('Note name')
    plt.ylabel('Number of timesteps played')
    plt.savefig('chorma_hist.png')

def plot_single_song_histogram(filename):
    song = pickle.load(open(filename, 'rb')) 
    pianoroll = mf.get_piano_roll(song)
    pianoroll[pianoroll > 0] = 1
    hist = np.sum(pianoroll, axis=1)
    plt.figure()
    plt.bar(range(128), hist, width=1)
    plt.title('Histogram of notes for one song')
    plt.xlabel('Note MIDI index')
    plt.ylabel('Nmber of timesteps played')
    plt.savefig('single_song_hist.png')

def plot_piano_roll(filename):
    song = pickle.load(open(filename, 'rb')) 
    pianoroll = mf.get_piano_roll(song)
    pianoroll[pianoroll > 0] = 1
    plt.figure(figsize=(6, 5))
    plt.imshow(pianoroll[:,128:256], cmap=cm.Purples, origin='lower')
    plt.title('Piano roll representation')
    plt.xlabel('Timestep (16th note)')
    plt.ylabel('Note MIDI index')
    plt.savefig('pianoroll.png')

if __name__ == "__main__":
    #plot_most_common_chords()
    plot_most_common_scales()
    #plot_full_histogram()
    #plot_chroma_histogram()
    #plot_piano_roll('chord_generator_test\data\midi_mod\\0fab31e0b3aae984222693b85a7d980c.pickle')
