from keras.models import load_model
import numpy as np
import midi_functions as mf
import load_data as load
import os
import config as conf
import pretty_midi
import data_preparation
import _pickle as pickle

np.random.seed(2014)

num_steps = 16
model_file = os.path.join(conf.results_dir, 'models', 'epoch085.hdf5')

filenames = load.list_pickle_files('melodies')

print("Loading model...")
model = load_model(model_file)
print("Model loaded.")

for i in range(10):
    random_song = filenames[np.random.randint(0, len(filenames))]
    
    melody = pickle.load(open(random_song, 'rb'))
    melody[melody > 0] = 1
    
    chords_input = [[[[]]]]
    chords_input[0][0][0] = 92
    melody_input = np.zeros((1, 16, conf.num_notes))
    output = np.zeros(num_steps+1)
    output[0] = 92

    print("Generating chords for song {}".format(i))
    #output = np.zeros((1,num_steps,1))
    #output[0,0,0] = 37
    for j in range(num_steps):
        x_chords = np.zeros((j+1))
        x_melody = np.zeros((j+1, 16, conf.num_notes))
        x_chords = output[0:j+1]
        x_melody[:,:,:] = melody[8:9,:,:]
        prediction = model.predict([x_chords, x_melody])
        next_chord = np.random.choice(len(prediction[j][0]), p=prediction[j][0])
        output[j+1] = next_chord#np.argmax(prediction[j], axis=1)
        #melody_input[0,:,:] = melody[j,:]
        #prediction = model.predict([chords_input, melody_input])
        #output[0,j,0] = np.argmax(prediction, axis=2)
        #chords_input.append(np.zeros((1,1)))
        #chords_input[j,0,0] = output[0,j,0]

    print("Writing chords...")
    _, index_to_chord = data_preparation.get_chord_dict()
    output = output.flatten()
    chords = [index_to_chord[i] for i in output]

    pianoroll_melody = np.reshape(melody, (-1, conf.num_notes)).T
    melody_midi = mf.piano_roll_to_midi(pianoroll_melody)

    midi_path = os.path.join(conf.results_dir, 'chords_and_melodies', f'test{i:02d}.mid')
    chord_path = os.path.join(conf.results_dir, 'chords_and_melodies', f'test{i:02d}.txt')

    orig = pretty_midi.PrettyMIDI()
    melody_midi.write(midi_path)
    chord_file = open(chord_path, 'w')
    for chord in chords:
        chord_file.write(chord)
        chord_file.write('\n')
    chord_file.close()


print()
