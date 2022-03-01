from keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import config as conf
import math
import _pickle as pickle
import midi_functions as mf
import load_data as load
import pretty_midi
import os
import time
np.random.seed(2021)
class MusicGenerator:
    def __init__(self):
        self.chord_model = load_model(conf.model_filenames['chord_model'])
        self.chord_model.call = tf.function(self.chord_model.call)
        self.poly_model = load_model(conf.model_filenames['poly_model'])
        self.poly_model.call = tf.function(self.poly_model.call)
        _, self.chord_dict = load.get_chord_dict()
        
        self.melody = np.zeros((1, conf.num_steps, conf.num_notes))

        #chord_seq = [92, 92, 85, 53, 92, 92, 53, 85]

        self.chords = np.zeros((1, math.floor(conf.num_steps/conf.chord_interval), 1))#np.reshape(chord_seq, (1, math.floor(conf.num_steps/conf.chord_interval), 1))#
        self.chords[0,0] = 0
        self.chords_expanded = np.zeros((1, conf.num_steps, 1))#np.repeat(self.chords, conf.chord_interval, axis=1)#
        self.chords_expanded[0, :16] = 0

        self.counter = to_categorical(np.tile(range(conf.chord_interval), math.floor(conf.num_steps/conf.chord_interval)), num_classes=conf.chord_interval).reshape((1, conf.num_steps, conf.chord_interval))
        self.current_timestep = 0
        self.start_of_sequence = True

    def chord_step(self):
        start_time = time.process_time()
        if self.start_of_sequence and math.floor(self.current_timestep/conf.chord_interval)+1 < self.chords.shape[1]:
            chord_step = math.floor(self.current_timestep/conf.chord_interval)
            conv_melody = np.reshape(self.melody, (1, -1, conf.chord_interval, conf.num_notes))
            prediction = self.chord_model([self.chords, conv_melody], training=False).numpy()
            next_chord = np.random.choice(len(prediction[0][chord_step]), p=prediction[0][chord_step])
            #next_chord = np.random.randint(0, 100)
            self.chords[0,chord_step+1,0] = next_chord
            self.chords_expanded[0, self.current_timestep+1:self.current_timestep+1+conf.chord_interval,0] = next_chord
            #print("Sequence start chord time: ", time.process_time() - start_time)
            print('next chord: ', self.chord_dict[next_chord])
        else:
            conv_melody = np.reshape(self.melody, (1, -1, conf.chord_interval, conf.num_notes))
            prediction = self.chord_model([self.chords, conv_melody], training=False).numpy()
            next_chord = np.random.choice(len(prediction[0][-1]), p=prediction[0][-1])
            #next_chord = np.random.randint(0, 100)
            self.chords[0] = np.roll(self.chords[0], -1)
            self.chords[0,-1,0] = next_chord
            self.chords_expanded[0] = np.roll(self.chords_expanded[0], -16)
            self.chords_expanded[0, -16:, 0] = next_chord
            #print("Sequence middle chord time: ", time.process_time() - start_time)
            print('next_chord:', self.chord_dict[next_chord])

    def step(self, timestep):
        start_time = time.process_time()
        if self.current_timestep % conf.chord_interval == conf.chord_interval - 1:
            self.chord_step()

        if self.start_of_sequence:

            self.melody[:, self.current_timestep] = timestep

            X = np.concatenate((self.melody, self.counter), axis=2)
            pred_start_time = time.process_time()
            #prediction = np.random.rand((204))
            prediction = self.poly_model([self.chords_expanded, X], training=False).numpy()[0][self.current_timestep]
            print('pred time: ', time.process_time() - pred_start_time)
            prediction = prediction
            prediction[prediction >= conf.threshold] = 100
            prediction[prediction < conf.threshold] = 0
            
            self.current_timestep += 1
            if self.current_timestep >= conf.num_steps:
                self.start_of_sequence = False

            #print('Sequence start poly time: ', time.process_time() - start_time)
            return prediction
        else:
            self.melody[0] = np.roll(self.melody[0], -1)
            self.melody[0,-1] = timestep
            self.chords_expanded[0] = np.roll(self.chords_expanded[0], -1)
            self.chords_expanded[0, -1] = self.chords[0, -1]

            X = np.concatenate((self.melody, self.counter), axis=2)
            #prediction = np.random.rand((204))
            pred_start_time = time.process_time()
            prediction = self.poly_model([self.chords_expanded, X], training=False)[0][-1]
            print('pred time', time.process_time() - pred_start_time)
            prediction[prediction >= conf.threshold] = 100
            prediction[prediction < conf.threshold] = 0
            self.current_timestep += 1

            #print('Sequence middle poly time: ', time.process_time() - start_time)
            return prediction

if __name__ == '__main__':
    generator = MusicGenerator()

    melody_filenames = pickle.load(open('val_filenames.pickle', 'rb'))

    random_song = "f"
    while not os.path.exists(os.path.join('chord_generator_test', 'data', 'melodies', random_song)):
        random_song = melody_filenames[np.random.randint(0, len(melody_filenames))+2]
    folder_name = os.path.basename(random_song).replace('.pickle', '')
    print(f"Generating accompaniment for {folder_name}")

    melody = pickle.load(open(os.path.join('chord_generator_test', 'data', 'melodies', random_song), 'rb'))
    melody[melody > 0] = 1

    melody_sum_step = np.sum(melody, axis=(1, 2))
    melody_sum_step[melody_sum_step > 0] = 1
    melody_start = np.argmax(melody_sum_step)
    melody = melody[melody_start:]
    melody = np.reshape(melody, (-1, 60))

    gen = np.zeros((melody.shape[0], (conf.num_notes*3)+24))
    for i in range(512):
        gen[i] = generator.step(melody[i])


    midi_output = mf.get_poly_output_as_midi(gen)

    # Get melody as MIDI
    pianoroll_melody = melody[melody_start*conf.chord_interval:(melody_start*conf.chord_interval)+conf.num_steps]
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
    
    print()
