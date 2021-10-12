import pretty_midi as pm
import os
import time
import numpy as np
import math
import automatic_music_accompaniment_master.code.utils as utils

from tensorflow.keras.preprocessing import sequence

np.random.seed(2016)

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

def searchName(keys, strings):
    for s in strings:
        for k in keys:
            if k.lower() in s.lower():
                return True
    return False

def parseMIDI_manual(midi_file_path):
    number_of_instruments = 5
    try:
        midi_data = pm.PrettyMIDI(midi_file_path)
    except:
        print("Could not load file {}".format(midi_file_path))
        return False
    
    end_time = midi_data.get_end_time()
    number_of_ts = utils.time_to_t(midi_data, end_time)

    data = np.zeros((number_of_instruments, number_of_ts, utils.NUMBER_FEATURES), dtype=np.bool)
    data[:, :, utils.NUMBER_FEATURES - 2] = 1 # rest
    for t in range(number_of_ts):
        if t % 4 == 0:
            data[:, t, utils.NUMBER_FEATURES - 1] = 1

    for inst in midi_data.instruments:
        onsets = []
        instNum = None
        inst_name = pm.program_to_instrument_name(inst.program)
        track_name = inst.name
        if inst.is_drum:
            instNum = 0
        elif searchName(["bass"], [inst_name, track_name]):
            instnum = 1
        elif searchName(["guit"], [inst_name, track_name]):
            instNum = 2
        elif searchName(["pian", "key", "organ"], [inst_name, track_name]):
            instNum = 3
        elif searchName(["voc", "voi", "choir"], [inst_name, track_name]):
            instNum = 4
        if instNum is None:
            break

        for note in inst.notes:
            start_t = utils.time_to_t(midi_data, note.start)
            end_t = utils.time_to_t(midi_data, note.end)
            if start_t < end_t:
                pitch = note.pitch
                data[instNum, start_t, pitch] = 1 # Note onset
                data[instNum, start_t + 1 : end_t, utils.NUMBER_FEATURES - 3] = 1 # sustain
                data[instNum, start_t : end_t, utils.NUMBER_FEATURES - 2] = 0 # rest 
                onsets.append(start_t)
        for onset in onsets:
            data[instNum, onset, utils.NUMBER_FEATURES - 3] = 0 # not sustain
        
    return data

    

def getSongs_manual(amount=0):
    files = listFiles()

    midi_x = []
    its = len(files)
    selected = []
    if amount > 0:
        its = amount
    startTime = time.time()
    for i in range(its):
        while True:
            rand = np.random.randint(0, len(files))
            if rand not in selected:
                break
        selected.append(rand)
        file = files[rand]

        data = parseMIDI_manual(file)
        if data is not False:
            midi_x.append(data)

    endTime = time.time()
    print("Loaded data in {} seconds.".format(endTime - startTime))
    
    return midi_x

def generate_datafiles(amount=0):
    files = listFiles()

    print("Starting data generation for", len(files), "MIDI files")
    its = len(files)
    selected = []
    if amount > 0:
        its = amount

    for i in range(its):
        if i % 100 == 0:
            print("Progress:", i, "files written")
        while True:
            rand = np.random.randint(0, len(files))
            if rand not in selected:
                break
        selected.append(rand)
        file = files[rand]

        data = parseMIDI_manual(file)
        if data is not False:
            path = 'zData{}.npy'.format(i)
            np.save(path, data)
    print("Done writing.")

def load_datafiles():
    data_path = os.path.join("cheriell_test", "data")
    data_files = [os.path.join(data_path, path) for path in os.listdir(data_path) if '.npy' in path]  
    midi_x = []
    startTime = time.time()
    for data_file in data_files:
        midi_x.append(np.load(data_file)[:, :, :utils.NUMBER_FEATURES-3]) #midi_x.append(np.load(data_file)[:, :, 0:128]) to remove rest, sustain and onset
    endTime = time.time()
    print("Loaded data in {} seconds.".format(endTime - startTime))
    train_count = math.floor((len(midi_x) / 100) * 80)
    train = midi_x[0:train_count]
    val = midi_x[train_count:]
    return train, val

#data = load_datafiles()
#generate_datafiles()
#print()