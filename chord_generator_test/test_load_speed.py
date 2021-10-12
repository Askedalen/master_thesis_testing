from re import L
import parse_midi
import numpy
import pretty_midi
import os
import _pickle as pickle
import time

midi_dir = os.path.join('chord_generator_test', 'data', 'test', 'midi')
piano_roll_dir = os.path.join('chord_generator_test', 'data', 'test', 'pianoroll')

def generate_test_files(amount):
    files = parse_midi.listFiles()
    print('Generating files...')
    for i in range(amount):
        file = files[i]
        try:
            midi_data = pretty_midi.PrettyMIDI(file)
        except:
            print("Could not load file {}".format(file))
            continue
        
        piano_roll = parse_midi.get_piano_roll(midi_data)

        filename = os.path.basename(file) + ".pickle"
        pickle.dump(midi_data, open(os.path.join(midi_dir, filename), 'wb'))
        pickle.dump(piano_roll, open(os.path.join(piano_roll_dir, filename), 'wb'))
    print('Done generating...')

def test_load_speed():
    midi_files = [os.path.join(midi_dir, path) for path in os.listdir(midi_dir) if '.pickle' in path]  
    piano_roll_files = [os.path.join(piano_roll_dir, path) for path in os.listdir(piano_roll_dir) if '.pickle' in path]  
    
    print('Starting to load MIDI data...')
    midi_start_time = time.time()
    midi_data = []
    for file in midi_files:
        midi_data.append(pickle.load(open(file, 'rb')))
    midi_end_time = time.time()
    print('Finished loading MIDI data...')
    print('Converting MIDI to piano rolls')
    
    midi_piano_rolls = []
    for song in midi_data:
        midi_piano_rolls.append(parse_midi.get_piano_roll(song))
    midi_pr_end_time = time.time()
    print('Finished extracting piano rolls from MIDI...')
    
    print('Starting to load piano roll data...')
    pr_start_time = time.time()
    piano_rolls = []
    for file in piano_roll_files:
        piano_rolls.append(pickle.load(open(file, 'rb')))
    pr_end_time = time.time()
    print('Finished loading piano roll data...')

    print()
    print('Times:')
    print('MIDI: ', midi_end_time - midi_start_time)
    print('MIDI with conversion to piano roll: ', midi_pr_end_time - midi_start_time)
    print('Piano rolls: ', pr_end_time - pr_start_time)
    print('midi_data=', len(midi_data), ',midi_piano_rolls=', len(midi_piano_rolls), ',piano_rolls=', len(piano_rolls))

def test_load_speed2():
    pickle_files = [os.path.join(midi_dir, path) for path in os.listdir(midi_dir) if '.pickle' in path]  
    midi_raw_files = parse_midi.listFiles()
    
    print('Starting to load pickle data...')
    pickle_start_time = time.time()
    pickle_data = []
    for file in pickle_files:
        pickle_data.append(pickle.load(open(file, 'rb')))
    pickle_end_time = time.time()
    print('Finished loading pickle data...')
    
    print('Starting to load MIDI data...')
    midi_start_time = time.time()
    midi_data = []
    for i in range(len(pickle_files)):
        file = midi_raw_files[i]
        try:
            midi_data.append(pretty_midi.PrettyMIDI(file))
        except:
            print("Could not load file {}".format(file))
            continue
    midi_end_time = time.time()
    print('Finished loading MIDI data...')

    print()
    print('Times:')
    print('Pickle: ', pickle_end_time - pickle_start_time)
    print('MIDI: ', midi_end_time - midi_start_time)
    print('pickle_data=', len(pickle_data), ',midi_data=', len(midi_data),)


if __name__ == '__main__':
    #generate_test_files(1000)
    test_load_speed()
    #test_load_speed2()