import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Activation, Embedding, Input, TimeDistributed, Conv1D, Conv2D
from tensorflow.keras.layers import Concatenate, add, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import reshape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.core import Reshape
import load_data as load
from generator import chord_data_generator, poly_data_generator, count_steps
import math
import time
import config as conf

class FullModel:
    def __init__(self, chord_config, poly_config):
        self.chord_config = chord_config
        self.poly_config = poly_config

    def train_chord_model(self, train_x, train_y, val_x, val_y):
        best_model = None
        self.chord_model = best_model
    
    def train_poly_model(self, train_x, train_y, val_x, val_y):
        best_model = None
        self.poly_model = best_model

if __name__ == "__main__":
    chord_config = {'lstm_size':512,
                    'batch_size':32,
                    'val_batch_size':32,
                    'learning_rate':0.00001,
                    'embedding_size':10,
                    'vocabulary':100,
                    'max_steps':8,
                    'chord_interval':16,
                    'epochs':100}

    poly_config = {'lstm_size':512,
                   'batch_size':128,
                   'val_batch_size':128,
                   'learning_rate':0.0001,
                   'vocabulary':(conf.num_notes*3)+24,
                   'max_steps':128,
                   'epochs':100}
    model = FullModel(chord_config, poly_config)

    train_filenames, val_filenames = load.get_trainval_filenames(rand_data=conf.random_data)
    
    chord_training_generator = chord_data_generator(train_filenames, infinite=False, **chord_config)
    chord_val_generator = chord_data_generator(val_filenames, infinite=False, **chord_config)
    
    chord_training_x = []
    chord_training_y = []
    for x, y in chord_training_generator:
        chord_training_x.append(x)
        chord_training_y.append(y)

    chord_val_x = []
    chord_val_y = []
    for x, y in chord_val_generator:
        chord_val_x.append(x)
        chord_val_y.append(y)

    model.train_chord_model(chord_training_x, chord_training_y, chord_val_x, chord_val_y)

    best_chord_model = model.best_chord_model
    chord_embedding = load.ChordEmbedding(best_chord_model)

    poly_training_generator = poly_data_generator(train_filenames, chord_embedding, infinite=False, **poly_config)
    poly_val_generator = poly_data_generator(val_filenames, chord_embedding, infinite=False, **poly_config)

    poly_training_x = []
    poly_training_y = []
    for x, y in poly_training_generator:
        poly_training_x.append(x)
        poly_training_y.append(y)

    poly_val_x = []
    poly_val_y = []
    for x, y in poly_val_generator:
        poly_val_x.append(x)
        poly_val_y.append(y)