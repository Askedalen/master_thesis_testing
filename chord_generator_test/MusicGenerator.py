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
np.random.seed(2021)

def get_chord_dict():
    if os.path.exists('chord_dict.pickle'):
        chord_dict = pickle.load(open('chord_dict.pickle', 'rb'))
    else:
        return None
    chord_to_index = chord_dict
    index_to_chord = dict((v,k) for k,v in chord_dict.items())
    return chord_to_index, index_to_chord

class MusicGenerator:
    def __init__(self, chord_model=None, poly_model=None, binary=False, threshold=0.25):
        if chord_model is None and poly_model is None:
            self.chord_model = load_model(conf.model_filenames['chord_model'])
            self.poly_model = load_model(conf.model_filenames['poly_model'])
        else:
            self.chord_model = chord_model
            self.poly_model = poly_model

        self.binary=binary
        self.threshold=threshold

        self.chord_model.call = tf.function(self.chord_model.call)
        self.poly_model.call = tf.function(self.poly_model.call)
        _, self.chord_dict = get_chord_dict()

        self.reset()
        
    def reset(self):
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
            #print('next chord: ', self.chord_dict[next_chord])
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
            #print('next_chord:', self.chord_dict[next_chord])

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
            #print('pred time: ', time.process_time() - pred_start_time)
            prediction = prediction
            #print(np.max(prediction), np.min(prediction), np.mean(prediction))
            if self.binary:
                prediction[prediction >= self.threshold] = 1
            else:
                prediction[prediction >= self.threshold] = 100
            prediction[prediction < self.threshold] = 0
            
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
            prediction = self.poly_model([self.chords_expanded, X], training=False).numpy()[0][-1]
            #print('pred time', time.process_time() - pred_start_time)
            if self.binary:
                prediction[prediction >= self.threshold] = 1
            else:
                prediction[prediction >= self.threshold] = 100
            prediction[prediction < self.threshold] = 0
            self.current_timestep += 1

            #print('Sequence middle poly time: ', time.process_time() - start_time)
            return prediction

class BaselineMusicGenerator:
    def __init__(self, model, binary=False, threshold=0.25):
        self.model = model
        self.model.call = tf.function(self.model.call)
        self.binary=binary
        self.threshold=threshold
        self.reset()
        
    def reset(self):
        self.melody = np.zeros((1, conf.num_steps, conf.num_notes))

        self.counter = to_categorical(np.tile(range(conf.chord_interval), math.floor(conf.num_steps/conf.chord_interval)), num_classes=conf.chord_interval).reshape((1, conf.num_steps, conf.chord_interval))
        self.current_timestep = 0
        self.start_of_sequence = True

    def step(self, timestep):
        start_time = time.process_time()
        if self.start_of_sequence:
            self.melody[:, self.current_timestep] = timestep
            X = np.concatenate((self.melody, self.counter), axis=2)
            pred_start_time = time.process_time()
            #prediction = np.random.rand((204))
            prediction = self.model(X, training=False).numpy()[0][self.current_timestep]
            #print('pred time: ', time.process_time() - pred_start_time)
            prediction = prediction
            if self.binary:
                prediction[prediction >= self.threshold] = 1
            else:
                prediction[prediction >= self.threshold] = 100
            prediction[prediction < self.threshold] = 0
            
            self.current_timestep += 1
            if self.current_timestep >= conf.num_steps:
                self.start_of_sequence = False

            #print('Sequence start poly time: ', time.process_time() - start_time)
            return prediction
        else:
            self.melody[0] = np.roll(self.melody[0], -1)
            self.melody[0,-1] = timestep

            X = np.concatenate((self.melody, self.counter), axis=2)
            #prediction = np.random.rand((204))
            pred_start_time = time.process_time()
            prediction = self.model(X, training=False).numpy()[0][-1]
            #print('pred time', time.process_time() - pred_start_time)
            if self.binary:
                prediction[prediction >= self.threshold] = 1
            else:
                prediction[prediction >= self.threshold] = 100
            prediction[prediction < self.threshold] = 0
            self.current_timestep += 1

            #print('Sequence middle poly time: ', time.process_time() - start_time)
            return prediction

if __name__ == '__main__':
    pass
