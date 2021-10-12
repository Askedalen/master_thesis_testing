import pretty_midi as pm
import os
import time
import numpy as np
import math

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

    drum_tracks = []
    bass_tracks = []
    piano_tracks = []
    guitar_tracks = []
    vocal_tracks = []

    for instrument in midi_data.instruments:
        notes = instrument.notes 
        inst_name = pm.program_to_instrument_name(instrument.program)
        track_name = instrument.name
        if instrument.is_drum:
            drum_tracks += notes
        elif searchName(["bass"], [inst_name, track_name]):
            bass_tracks += notes
        elif searchName(["guit"], [inst_name, track_name]):
            guitar_tracks += notes
        elif searchName(["pian", "key", "organ"], [inst_name, track_name]):
            piano_tracks += notes
        elif searchName(["voc", "voi", "choir"], [inst_name, track_name]):
            vocal_tracks += notes

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

    pr_x = midi_x #np.concatenate(midi_x, axis=1)
    pr_y = midi_y #np.concatenate(midi_y, axis=1)

    endTime = time.time()
    print("Loaded data in {} seconds.".format(endTime - startTime))
    return pr_x, pr_y

def createModelInput(piano_rolls, sequence_length=50):
    x = []
    y = []
    for song in piano_rolls:
        pos = 0
        while pos+sequence_length < song.shape[0]:
            sample = np.array(song[pos:pos+sequence_length])
            x.append(sample)
            y.append(song[pos+sequence_length])
            pos += 1
    return np.array(x), np.array(y)

data = getSongs(1)
