from keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import config as conf
import math
np.random.seed(2021)
class MusicGenerator:
    def __init__(self):
        self.chord_model = load_model(conf.model_filenames['chord_model'])
        self.chord_model.load_weights(conf.model_filenames['chord_weights'])

        self.poly_model = load_model(conf.model_filenames['poly_model'])
        self.poly_model.load_weights(conf.model_filenames['poly_weights'])
        
        self.melody = np.zeros((1, conf.num_steps, conf.num_notes))

        chord_seq = [92, 92, 85, 53, 92, 92, 53, 85]

        self.chords = np.reshape(chord_seq, (1, math.floor(conf.num_steps/conf.chord_interval), 1))#np.zeros((1, math.floor(conf.num_steps/conf.chord_interval), 1))
        self.chords[0,0] = 92
        self.chords_expanded = np.repeat(self.chords, conf.chord_interval, axis=1)#np.zeros((1, conf.num_steps, 1))

        self.counter = to_categorical(np.tile(range(conf.chord_interval), math.floor(conf.num_steps/conf.chord_interval)), num_classes=conf.chord_interval).reshape((1, conf.num_steps, conf.chord_interval))
        self.current_timestep = 0
        self.start_of_sequence = False

    def chord_step(self):
        if self.start_of_sequence:
            chord_step = math.floor(self.current_timestep/conf.chord_interval)
            conv_melody = np.reshape(self.melody, (1, -1, conf.chord_interval, conf.num_notes))
            prediction = self.chord_model.predict([self.chords, conv_melody])
            next_chord = np.random.choice(len(prediction[chord_step][0]), p=prediction[chord_step][0])
            self.chords[0,chord_step+1,0] = next_chord
            self.chords_expanded[0, self.current_timestep:self.current_timestep+conf.chord_interval,0] = next_chord
        else:
            conv_melody = np.reshape(self.melody, (1, -1, conf.chord_interval, conf.num_notes))
            prediction = self.chord_model.predict([self.chords, conv_melody])
            next_chord = np.random.choice(len(prediction[0][-1]), p=prediction[0][-1])
            self.chords[0] = np.roll(self.chords[0], -1)
            self.chords[0,-1,0] = next_chord

    def step(self, timestep):
        if self.current_timestep % conf.chord_interval == conf.chord_interval - 1:
            self.chord_step()

        if self.start_of_sequence:

            self.melody[:, self.current_timestep] = timestep

            X = np.concatenate((self.melody, self.counter), axis=2)
            prediction = self.poly_model.predict([self.chords_expanded, X])[0][self.current_timestep]
            """ prediction[prediction >= conf.threshold] = 100
            prediction[prediction < conf.threshold] = 0 """
            
            self.current_timestep += 1
            if self.current_timestep >= conf.num_steps:
                self.start_of_sequence = False

            return prediction
        else:
            self.melody[0] = np.roll(self.melody[0], -1)
            self.melody[0,-1] = timestep
            self.chords_expanded[0] = np.roll(self.chords_expanded[0], -1)
            self.chords_expanded[0, -1] = self.chords[0, -1]

            
            X = np.concatenate((self.melody, self.counter), axis=2)
            prediction = self.poly_model.predict([self.chords_expanded, X])
            self.current_timestep += 1
            return prediction[0,-1]

if __name__ == '__main__':
    generator = MusicGenerator()
    test = np.zeros((128, conf.num_notes))
    starting_note = 60 - conf.pr_start_idx

    gen = np.zeros((128, (conf.num_notes*3)+24))
    for i in range(128):
        gen[i] = generator.step(test[i])
    print()
