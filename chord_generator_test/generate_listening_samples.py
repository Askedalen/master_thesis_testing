from keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import config as conf
import math
import _pickle as pickle
import midi_functions as mf
import pretty_midi
import os
import time
from MusicGenerator import MusicGenerator

if conf.testing:
    chord_model_filename = "chord_generator_test\\results\\tests\\full_model\models\chord_model.pb"
    poly_model_filename = "chord_generator_test\\results\\tests\\full_model\models\poly_model.pb"
    baseline_model_filename = "chord_generator_test\\results\\tests\\baseline\\baseline_d220411_t2140/model.pb"
else:
    chord_model_filename = "results/tests/d220417_t1436/models/chord_model.pb"
    poly_model_filename = "results/tests/d220417_t1938/models/poly_model.pb"
    baseline_model_filename = "results/tests/baseline_d220411_t2002/model.pb"

chord_model = load_model(chord_model_filename)
poly_model = load_model(poly_model_filename)

generator = MusicGenerator(chord_model=chord_model, poly_model=poly_model)

melody_filenames = pickle.load(open('test_filenames.pickle', 'rb'))

melody_filenames = melody_filenames[:10]

for random_song in melody_filenames:
    folder_name = os.path.basename(random_song).replace('.pickle', '')
    print(f"Generating accompaniment for {folder_name}")
    
    melody = pickle.load(open(os.path.join(conf.melody_dir, random_song), 'rb'))
    melody[melody > 0] = 1

    melody_sum_step = np.sum(melody, axis=(1, 2))
    melody_sum_step[melody_sum_step > 0] = 1
    melody_start = np.argmax(melody_sum_step)
    melody = melody[melody_start:]
    melody = np.reshape(melody, (-1, 60))

    gen = np.zeros((melody.shape[0], (conf.num_notes*3)+24))
    for i in range(len(melody)):
        gen[i] = generator.step(melody[i])


    midi_output = mf.get_poly_output_as_midi(gen)

    # Get melody as MIDI
    pianoroll_melody = melody.astype(float)
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
    
    print()