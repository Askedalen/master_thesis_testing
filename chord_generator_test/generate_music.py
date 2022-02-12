from keras.models import load_model
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
import midi_functions as mf
import load_data as load
import os
import config as conf
import math
import pretty_midi
import data_preparation
import _pickle as pickle


num_steps = 128
chord_interval = 16
number_of_melodies = 10
threshold = 0.25

chord_best_epoch = 100
poly_best_epoch = 99

chord_model_filename = os.path.join(conf.chord_model_dir, f'epoch{chord_best_epoch:03d}.hdf5')
poly_model_filename = os.path.join(conf.poly_model_dir, f'epoch{poly_best_epoch:03d}.hdf5')

melody_filenames = load.list_pickle_files('melodies')#pickle.load(open('val_filenames.pickle', 'rb'))

""" chord_model = load_model('pyo_testing/models/chord_best.hdf5', compile=False)
chord_model2 = load_model('pyo_testing/models/chord_best', compile=False)
chord_model3 = load_model('pyo_testing/models/chord_model.pb', compile=False)
chord_weights = chord_model.get_weights()
chord2_weights = chord_model2.get_weights()
chord3_weights = chord_model3.get_weights()
chord_model3.set_weights(chord_weights)
test = chord_model3.get_weights()
chord_model3.set_weights(chord2_weights)
test2 = chord_model3.get_weights()
chord_model3.load_weights('pyo_testing/models/chord_weights.pb')
test3 = chord_model3.get_weights() """

chord_model = load_model('pyo_testing/models/chord_model.pb', compile=False)
chord_model.load_weights('pyo_testing/models/chord_weights.pb')

poly_model = load_model('pyo_testing/models/poly_model.pb', compile=False)
poly_model.load_weights('pyo_testing/models/poly_weights.pb')

chord_input = np.zeros((1, 8, 1))
poly_input = np.zeros((1, 8, 16, 60))
test_chord_model = load_model('pyo_testing/models/test_chord_model.hdf5')
chord_pred_1 = test_chord_model.predict([chord_input, poly_input])
chord_pred_2 = chord_model.predict([chord_input, poly_input])

test_poly_model = load_model('pyo_testing/models/test_poly_model.hdf5')

for i in range(number_of_melodies):
    random_song = melody_filenames[np.random.randint(0, len(melody_filenames))]
    if not os.path.exists(random_song):
        print('not exist')
        continue
    folder_name = os.path.basename(random_song).replace('.pickle', '')
    print(f"Generating accompaniment for {folder_name}")

    melody = pickle.load(open(random_song, 'rb'))
    melody[melody > 0] = 1

    melody_sum_step = np.sum(melody, axis=(1, 2))
    melody_sum_step[melody_sum_step > 0] = 1
    melody_start = np.argmax(melody_sum_step)

    chord_output = np.zeros((math.floor(num_steps/chord_interval))+1)
    chord_output[0] = 92

    for j in range(math.floor(num_steps/chord_interval)):
        x_chords = np.zeros((1, math.floor(num_steps/chord_interval)), dtype='float32')
        x_chords[0,0:j+1] = chord_output[0:j+1]

        x_melody = np.zeros((1, math.floor(num_steps/chord_interval), chord_interval, conf.num_notes), dtype='float32')
        x_melody[0,0:j+1,:,:] = melody[melody_start:melody_start+j+1, :, :]
        x_melody = np.float32(x_melody)
        
        prediction = chord_model.predict([x_chords, x_melody])[0]
        next_chord = np.random.choice(len(prediction[j]), p=prediction[j])
        chord_output[j+1] = next_chord

    model_output = np.zeros((num_steps, (conf.num_notes*3)+24))
    melody_expanded = np.reshape(melody, (-1, conf.num_notes))
    chords_expanded = np.repeat(chord_output, chord_interval, axis=0)
    counter = to_categorical(np.tile(range(chord_interval), math.floor(num_steps/chord_interval)), num_classes=chord_interval)
    
    for j in range(num_steps):
        x_chords = np.zeros((1, num_steps, 1))
        x_melody = np.zeros((1, num_steps, conf.num_notes))
        x_counter = np.zeros((1, num_steps, chord_interval))

        x_chords[0,:j+1,0] = chords_expanded[0:j+1]
        x_melody[0,:j+1] = melody_expanded[melody_start*chord_interval:(melody_start*chord_interval)+j+1, :]
        x_counter[0,:j+1] = counter[0:j+1]

        x = np.concatenate((x_melody, x_counter), axis=2)

        prediction = poly_model.predict([x_chords, x])
        next_step = prediction[0][j]

        model_output[j,:] = next_step[:]

    _, index_to_chord = load.get_chord_dict()
    chords = [index_to_chord[int(c)] for c in chord_output]

    model_output[model_output >= threshold] = 100#1 for binary, 100 is velocity
    model_output[model_output < threshold] = 0

    midi_output = mf.get_poly_output_as_midi(model_output)

    # Get melody as MIDI
    pianoroll_melody = melody_expanded[melody_start*chord_interval:(melody_start*chord_interval)+num_steps]
    pianoroll_melody = pianoroll_melody.astype(float)
    pianoroll_melody[pianoroll_melody > 0] = 100
    melody_midi = mf.piano_roll_to_midi(pianoroll_melody, 'melody', program=66)
    melody_midi_solo = pretty_midi.PrettyMIDI()
    melody_midi_solo.instruments.append(melody_midi)

    os.mkdir(os.path.join(conf.music_gen_dir, folder_name))

    poly_midi_path = os.path.join(conf.music_gen_dir, folder_name, 'only_comp.mid')
    poly_melody_path = os.path.join(conf.music_gen_dir, folder_name, 'comp_and_melody.mid')
    melody_solo_path = os.path.join(conf.music_gen_dir, folder_name, 'only_melody.mid')
    chord_path = os.path.join(conf.music_gen_dir, folder_name, 'chord_sequence.txt')

    midi_output.write(poly_midi_path)
    midi_output.instruments.append(melody_midi)
    midi_output.write(poly_melody_path)
    melody_midi_solo.write(melody_solo_path)
    
    chord_file = open(chord_path, 'w')
    for chord in chords:
        chord_file.write(chord)
        chord_file.write('\n')
    chord_file.close()
    