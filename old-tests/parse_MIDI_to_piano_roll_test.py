import pretty_midi as pm
import os
import time
import numpy as np
import math
import automatic_music_accompaniment_master.code.utils as utils

from tensorflow.keras.preprocessing import sequence

np.random.seed(2016)

def searchName(keys, strings):
    for s in strings:
        for k in keys:
            if k.lower() in s.lower():
                return True
    return False


def parseMIDI(midi_file_path):
    try:
        midi_data = pm.PrettyMIDI(midi_file_path)
    except:
        print("Could not load file {}".format(midi_file_path))
        return False

    initial_tempo = midi_data.get_tempo_changes()[1][0]
    beat_length_seconds = 60 / initial_tempo # Get length of each beat in seconds
    sampling_freq = 1 / (beat_length_seconds / 8) # Calculate sampling frequency to get 32nd notes

    piano_roll_shape = midi_data.get_piano_roll().shape

    drum_tracks = []
    bass_tracks = []
    piano_tracks = []
    guitar_tracks = []
    vocal_tracks = []

    for instrument in midi_data.instruments:
        piano_roll = instrument.get_piano_roll(fs=sampling_freq) # instrument.get_chroma() for chromagram 
        inst_name = pm.program_to_instrument_name(instrument.program)
        track_name = instrument.name
        if instrument.is_drum:
            drum_tracks.append(piano_roll)
        elif searchName(["bass"], [inst_name, track_name]):
            bass_tracks.append(piano_roll)
        elif searchName(["guit"], [inst_name, track_name]):
            guitar_tracks.append(piano_roll)
        elif searchName(["pian", "key", "organ"], [inst_name, track_name]):
            piano_tracks.append(piano_roll)
        elif searchName(["voc", "voi", "choir"], [inst_name, track_name]):
            vocal_tracks.append(piano_roll)

    #Create combined roll for all drum tracks
    drum_roll = np.zeros(piano_roll_shape)
    for c in drum_tracks:
        drum_roll[:, :c.shape[1]] += c

    #Create combined roll for all bass tracks
    bass_roll = np.zeros(piano_roll_shape)
    for c in bass_tracks:
        bass_roll[:, :c.shape[1]] += c

    #Create combined roll for all piano tracks
    piano_roll = np.zeros(piano_roll_shape)
    for c in piano_tracks:
        piano_roll[:, :c.shape[1]] += c

    #Create combined roll for all guitar tracks
    guitar_roll = np.zeros(piano_roll_shape)
    for c in guitar_tracks:
        guitar_roll[:, :c.shape[1]] += c

    #Create concatinated piano roll
    combined_roll = np.concatenate((drum_roll, bass_roll, piano_roll, guitar_roll))

    playing_tracks = vocal_tracks + guitar_tracks + piano_tracks
    playing_tracks_reshaped = []
    for p in playing_tracks:
        if p.shape != piano_roll_shape:
            new = np.zeros(piano_roll_shape)
            new[:, :p.shape[1]] += p
            playing_tracks_reshaped.append(new)
        else:
            playing_tracks_reshaped.append(p)
    
    return playing_tracks_reshaped, combined_roll

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

def getSongs(amount=0):
    files = listFiles()

    midi_x = []
    midi_y = []
    i = 0
    its = len(files)
    if amount > 0:
        its = amount
    startTime = time.time()
    for i in range(its):
        rand = math.floor(np.random.uniform(0, len(files)))
        file = files[rand]
        data = parseMIDI(file)
        if data is not False:
            xs, y = data
            """ for x in xs:
                midi_x.append(x)
                midi_y.append(y) """
            if len(xs) > 0:
                if len(xs) == 1:
                    midi_x.append(xs[0])
                else:
                    midi_x.append(xs[np.random.randint(0, len(xs)-1)])
                midi_y.append(y)

    pr_x = np.concatenate(midi_x, axis=1)
    pr_y = np.concatenate(midi_y, axis=1)

    endTime = time.time()
    print("Loaded data in {} seconds.".format(endTime - startTime))
    return pr_x, pr_y



#data = getSongs_manual(10)
