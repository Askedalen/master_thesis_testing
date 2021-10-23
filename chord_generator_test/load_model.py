from keras.models import load_model
import numpy as np
import parse_midi
import midi_functions as mf
import load_data as load
import chord_extraction
from config import *
import pretty_midi
import _pickle as pickle

np.random.seed(2019)

num_steps = 500
model_file = os.path.join(results_dir, 'models', 'epoch018.hdf5')

filenames = load.list_pickle_files('melodies')

print("Loading model...")
model = load_model(model_file)
print("Model loaded.")

for i in range(10):
    random_song = filenames[np.random.randint(0, len(filenames))]
    random_song_midi = pickle.load(open(random_song, 'rb'))
    random_song_midi[random_song_midi > 0] = 1
    
    melody = random_song_midi.T
    melody_nonzero = np.nonzero(melody)[0]
    if melody_nonzero.shape[0] == 0:
        i -= 1
        continue

    starting_point = melody_nonzero[0]
    chords_input = np.zeros((1, 1, 1))
    chords_input[0][0][0] = 0
    melody_input = np.zeros((1, 1, 128))

    print("Generating chords for song {}".format(i))
    output = np.zeros((1,num_steps,1))
    output[0,:16,0] = 92
    for j in range(num_steps):
        melody_input[0,0,:] = melody[j+starting_point,:]
        prediction = model.predict([chords_input, melody_input])
        output[0,j,0] = np.argmax(prediction, axis=2)
        chords_input[0,0,0] = output[0,j,0]

    print("Writing chords...")
    _, index_to_chord = chord_extraction.get_chord_dict()
    output = output.flatten()
    chords = [index_to_chord[i] for i in output]

    melody_midi = mf.piano_roll_to_midi(melody.T)
    chords_4ths = chords[::4]

    midi_path = os.path.join(data_dir, 'test', 'chords_and_melodies', f'test{i:02d}.mid')
    chord_path = os.path.join(data_dir, 'test', 'chords_and_melodies', f'test{i:02d}.txt')

    orig = pretty_midi.PrettyMIDI()
    melody_midi.write(midi_path)
    chord_file = open(chord_path, 'w')
    for chord in chords:
        chord_file.write(chord)
        chord_file.write('\n')
    chord_file.close()


print()
