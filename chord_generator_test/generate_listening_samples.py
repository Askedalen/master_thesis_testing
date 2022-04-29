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
from MusicGenerator import MusicGenerator, BaselineMusicGenerator

if conf.testing:
    chord_model_filename = "chord_generator_test\\results\\tests\\full_model\models\chord_model.pb"
    poly_model_filename = "chord_generator_test\\results\\tests\\full_model\models\poly_model.pb"
    baseline_model_filename = "chord_generator_test\\results\\tests\\baseline\\baseline_d220411_t2140/model.pb"
else:
    chord_model_filename = "results/tests/d220421_t1123/models/chord_model.pb"
    poly_model_filename = "results/tests/d220422_t1154/models/poly_model.pb"
    baseline_model_filename = "results/tests/baseline_d220419_t1610/model.pb"

chord_model = load_model(chord_model_filename)
poly_model = load_model(poly_model_filename)
#baseline_model = load_model(baseline_model_filename)

threshold = 0.08
#threshold = 0.18

generator = MusicGenerator(chord_model=chord_model, poly_model=poly_model, threshold=threshold)
#generator = BaselineMusicGenerator(baseline_model, threshold=threshold)

melody_filenames = pickle.load(open('test_filenames.pickle', 'rb'))

melody_filenames = melody_filenames[50:100]

#melody_filenames = ['33e5a8cf07044399f6b99635aee74244.pickle', '532c7ba6291d68c7803035f193a52b76.pickle', '8c8f35129e6f36d33bfa37c508b33c1e.pickle']

for random_song in melody_filenames:
    generator.reset()
    folder_name = os.path.basename(random_song).replace('.pickle', '')
    if os.path.exists(os.path.join(conf.music_gen_dir, 'baseline', folder_name)):
        print(f'{folder_name} already exists')
        continue
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

    os.mkdir(os.path.join(conf.music_gen_dir, 'baseline', folder_name))

    poly_midi_path = os.path.join(conf.music_gen_dir, 'baseline', folder_name, 'only_comp.mid')
    poly_melody_path = os.path.join(conf.music_gen_dir, 'baseline', folder_name, 'comp_and_melody.mid')
    melody_solo_path = os.path.join(conf.music_gen_dir, 'baseline', folder_name, 'only_melody.mid')
    chord_path = os.path.join(conf.music_gen_dir, 'baseline', folder_name, 'chord_sequence.txt')

    midi_output.write(poly_midi_path)
    midi_output.instruments.append(melody_midi)
    midi_output.write(poly_melody_path)
    melody_midi_solo.write(melody_solo_path)
    
    print()
