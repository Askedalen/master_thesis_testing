from config import *
import pretty_midi
import midi_functions as mf
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import _pickle as pickle


np.random.seed(2020)

def listFiles():
    files = []
    if os.path.exists('song_list.txt'):
        txt = open("song_list.txt", "r")
        for line in txt:
            files.append(line.replace('\n', ''))
        txt.close()
    else:
        txt = open("song_list.txt", "w")
        path = "lmd_matched"
        for r, d, f in os.walk(path):
            for file in f:
                if '.mid' in file:
                    files.append(os.path.join(r, file))
                    txt.write(os.path.join(r, file))
                    txt.write("\n")
        txt.close()
    return files

def midi_to_pickle(num_files=0):
    files = listFiles()
    its = len(files)
    random_start = 0
    if num_files > 0:
        its = num_files
        random_start = np.random.randint(0, len(files) - num_files)
    num_failed = 0

    start_time = time.time()
    for i in range(its):
        file = files[i + random_start]
        filename = os.path.basename(file).replace('.mid', '.pickle')
        midi_file = os.path.join(midi_dir, filename)
        combined_file = os.path.join(pianoroll_dir, filename)
        melody_file = os.path.join(melody_dir, filename)
        if os.path.exists(melody_file):
            continue

        try:
            midi_data = pretty_midi.PrettyMIDI(file)
        except:
            print("Could not load file {}".format(file))
            continue

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

        combined_roll = mf.get_piano_roll(midi_data_modulated)
        melody = mf.get_random_melody(midi_data_modulated)

        if melody is False:
            i -= 1
            num_failed += 1
            continue
        
        pickle.dump(midi_data_modulated, open(midi_file, 'wb'))
        pickle.dump(combined_roll, open(combined_file, 'wb'))
        pickle.dump(melody, open(melody_file, 'wb'))

        if i % 10 == 0:
            print("Finished {} songs".format(i))
        
    end_time = time.time()
    print(f"Finisned {its} songs in {end_time - start_time} seconds")

if __name__ == "__main__":
    midi_to_pickle(1000)
    #parse_midi()