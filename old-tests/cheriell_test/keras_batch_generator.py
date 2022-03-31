import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import numpy as np
import matplotlib.pyplot as plt
import automatic_music_accompaniment_master.code.utils as utils

NUMBER_FEATURES = 128#utils.NUMBER_FEATURES # 128 midi_notes + sustain + rest + beat_start
INSTRUMENTS = 5 # number of instruments in midifile

class KerasBatchGenerator(object):
    # This class is used for generating batches for training. Should be input to model.fit_generator()
    # Args:
    # data shape: (number of files)(instruments, timesteps, features)
    # num_steps: number of time steps in an unrolled model
    # batch_size: number of samples in a batch
    #vocabulary: number of categories in output
    #skip_step: number of steps to skip after each sample

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=3):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_index = 0 # index of t in current file
        self.file_index = 0 # index of current file
        self.skip_step = skip_step

    # Generator function called at the beginning of each batch during training
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, INSTRUMENTS * NUMBER_FEATURES), dtype=np.bool)
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary), dtype=np.bool)

        while True:
            for i in range(self.batch_size):
                if self.current_index + self.num_steps >= len(self.data[self.file_index][0]):
                    self.current_index = 0
                    self.file_index = (self.file_index + 1) % len(self.data)

                while len(self.data[self.file_index][0]) < self.num_steps:
                    self.file_index = (self.file_index + 1) % len(self.data)

                for inst_num in range(self.data[self.file_index].shape[0]):
                    x[i, :, inst_num*NUMBER_FEATURES : ((inst_num+1)*NUMBER_FEATURES)] = self.data[self.file_index][inst_num, self.current_index : self.current_index + self.num_steps, :]
                #x[i, :, :NUMBER_FEATURES] = self.data[self.file_index][0, self.current_index : self.current_index + self.num_steps, :]
                #x[i, 1:, NUMBER_FEATURES:] = self.data[self.file_index][1, self.current_index : self.current_index + self.num_steps - 1, :]
                y[i, :, :] = self.data[self.file_index][3, self.current_index : self.current_index + self.num_steps, :self.vocabulary]

                self.current_index += self.skip_step
            
            yield x, y

    def generate_test(self):
        #Used to generate sequences with input size 128 instead of 131
        x = np.zeros((self.batch_size, self.num_steps, NUMBER_FEATURES), dtype=np.bool)
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary), dtype=np.bool)

        while True:
            for i in range(self.batch_size):
                if self.current_index + self.num_steps >= len(self.data[self.file_index][0]):
                    self.current_index = 0
                    self.file_index = (self.file_index + 1) % len(self.data)

                while len(self.data[self.file_index][0]) < self.num_steps:
                    self.file_index = (self.file_index + 1) % len(self.data)

                for inst_num in range(self.data[self.file_index].shape[0]):
                    y[i, :, inst_num*NUMBER_FEATURES : ((inst_num+1)*NUMBER_FEATURES)] = self.data[self.file_index][inst_num, self.current_index : self.current_index + self.num_steps, :]
                #x[i, :, :NUMBER_FEATURES] = self.data[self.file_index][0, self.current_index : self.current_index + self.num_steps, :]
                #x[i, 1:, NUMBER_FEATURES:] = self.data[self.file_index][1, self.current_index : self.current_index + self.num_steps - 1, :]
                x[i, :, :] = self.data[self.file_index][3, self.current_index : self.current_index + self.num_steps, :self.vocabulary]

                self.current_index += self.skip_step
            
            yield x, y