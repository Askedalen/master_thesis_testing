from keras.models import load_model
import numpy as np
import parse_midi
import midi_functions as mf
import load_data as load
import chord_extraction
from config import *
import pretty_midi
import _pickle as pickle

np.random.seed(2021)

num_steps = 500
model_file = os.path.join(results_dir, 'models', 'epoch100.hdf5')

filenames = load.list_pickle_files('melodies')

print("Loading model...")
model = load_model(model_file)
print("Model loaded.")

for i in range(10):
    random_song = filenames[np.random.randint(0, len(filenames))]
    random_song_midi = pickle.load(open(random_song, 'rb'))
    
    melody = melody.T

    starting_point = np.nonzero(melody)[0][0]
    chords_input = np.zeros((1, 1, 1))
    melody_input = np.zeros((1, 1, 128))

    print("Generating chords for song {}".format(i))
    output = np.zeros((1,num_steps,1))
    for j in range(num_steps):
        melody_input[0,0,:] = melody[j+starting_point,:]
        prediction = model.predict([chords_input, melody_input])
        chords_input[0,0,0] = np.argmax(prediction, axis=2)
        output[0,j,0] = chords_input[0,0,0]

    print("Writing chords...")
    chord_dict = chord_extraction.get_chord_dict()
    chord_dict2 = dict((v,k) for k,v in chord_dict.items())
    output = output.flatten()
    chords = [chord_dict2[i] for i in output]

    melody_midi = mf.piano_roll_to_midi(melody.T)
    chords_4ths = chords[::4]

    orig_midi_path = os.path.join(data_dir, 'test', 'chords_and_melodies', f'orig{i:02d}.mid')
    midi_path = os.path.join(data_dir, 'test', 'chords_and_melodies', f'test{i:02d}.mid')
    chord_path = os.path.join(data_dir, 'test', 'chords_and_melodies', f'test{i:02d}.txt')

    orig = pretty_midi.PrettyMIDI()
    orig.instruments.append(orig_melody)
    orig.write(orig_midi_path)
    melody_midi.write(midi_path)
    chord_file = open(chord_path, 'w')
    for chord in chords_4ths:
        chord_file.write(chord)
        chord_file.write('\n')
    chord_file.close()


print()
