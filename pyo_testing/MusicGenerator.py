from keras.models import load_model
import numpy as np
import config as conf

class MusicGenerator:
    def __init__(self, ):
        self.chord_model = load_model(conf.model_filenames['chord_model'])
        self.chord_model.load_weights(conf.model_filenames['chord_weights'])

        self.poly_model = load_model(conf.model_filenames['poly_model'])
        self.poly_model.load_weights(conf.model_filenames['poly_weights'])
        
        self.melody = np.zeros((conf.num_steps+conf.chord_interval, conf.num_notes))
        self.current_timestep = 0
        self.start_of_sequence = True

    def step(self, timestep):
        if self.start_of_sequence:
            self.melody[self.current_timestep] = timestep
            self.current_timestep += 1
            if self.current_timestep >= (conf.num_steps+conf.chord_interval) - 1:
                self.start_of_sequence = False
        else:
            if self.current_timestep < conf.num_steps+conf.chord_interval:
                self.melody[-(16-self.current_timestep)] = timestep
                self.current_timestep += 1
            else:
                self.melody = np.roll(self.melody, -16)
                self.melody[-(16-self.current_timestep)] = timestep

