import os
from queue import Full
import numpy as np
import matplotlib.pyplot as plt
from six import python_2_unicode_compatible
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Activation, Embedding, Input, TimeDistributed, Conv1D, Conv2D
from tensorflow.keras.layers import Concatenate, add, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical, Progbar
from tensorflow.python.keras.backend import reshape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.backend import reshape, function
from tensorflow.python.keras.layers.core import Flatten, Reshape
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.initializers import Constant
from tensorflow.random import set_seed
import load_data as load
from generator import chord_data_generator, poly_data_generator, count_steps, embed_poly_chords
import math
import time
import datetime
import config as conf

np.random.seed(42)
set_seed(42)

def yield_data(data):
    while True:
        for [x1, x2], y in data:
            yield [x1, x2], y

if __name__ == '__main__':
    chord_config = {'batch_size':128,
                    'vocabulary':100,
                    'max_steps':8,
                    'num_notes':60,
                    'chord_interval':16,
                    'lstm_size':512,
                    'learning_rate':0.00001,
                    'embedding_size':10,
                    'epochs':100,
                    'verbose':0}

    poly_config = {'batch_size':256,
                    'vocabulary':(conf.num_notes*3)+24,
                    'max_steps':128,
                    'num_notes':60,
                    'chord_interval':16,
                    'lstm_size':1024,
                    'learning_rate':0.01,
                    'embedding_size':10,
                    'epochs':100,
                    'verbose':0}

    chord_optimizer = Adam(learning_rate=chord_config['learning_rate'])
    chord_loss = 'categorical_crossentropy'
    poly_optimizer = Adam(learning_rate=poly_config['learning_rate'])
    poly_loss = 'binary_crossentropy'

    chord_model = load_model('results/tests/d220209_t1137_cbs128_clr1e-05_cls512_pbs256_plr0.01_pls1024/models/chord_model.pb')
    #chord_model.load_weights('results/tests/d220207_t1255_cbs128_clr1e-05_cls512_pbs256_plr0.01_pls1024/models/chord_weights.pb')
    chord_model.compile(chord_optimizer, chord_loss, metrics=['categorical_accuracy'])

    poly_model = load_model('results/tests/d220209_t1137_cbs128_clr1e-05_cls512_pbs256_plr0.01_pls1024/models/poly_model.pb')
    #poly_model.load_weights('results/tests/d220207_t1255_cbs128_clr1e-05_cls512_pbs256_plr0.01_pls1024/models/poly_weights.pb')
    poly_model.compile(poly_optimizer, poly_loss, metrics=['categorical_accuracy'])

    chord_config = {'batch_size':128,
                    'vocabulary':100,
                    'max_steps':8,
                    'num_notes':60,
                    'chord_interval':16,
                    'lstm_size':512,
                    'learning_rate':0.00001,
                    'embedding_size':10,
                    'epochs':100,
                    'verbose':0}

    poly_config = {'batch_size':256,
                    'vocabulary':(conf.num_notes*3)+24,
                    'max_steps':128,
                    'num_notes':60,
                    'chord_interval':16,
                    'lstm_size':1024,
                    'learning_rate':0.01,
                    'embedding_size':10,
                    'epochs':100,
                    'verbose':0}

    train_filenames, val_filenames = load.get_trainval_filenames(rand_data=conf.random_data)
    train_filenames = train_filenames[:100]
    val_filenames = val_filenames[:100]
    chord_config = {'batch_size':128,
                    'vocabulary':100,
                    'max_steps':8,
                    'num_notes':60,
                    'chord_interval':16}

    poly_config = {'batch_size':256,
                    'vocabulary':(conf.num_notes*3)+24,
                    'max_steps':128,
                    'num_notes':60,
                    'chord_interval':16}
    # Get chord data
    print('Loading data...')
    #print()
    #print('Loading chord data...')
    chord_training_generator = chord_data_generator(train_filenames, infinite=False, **chord_config)
    chord_val_generator = chord_data_generator(val_filenames, infinite=False, **chord_config)
    chord_training_data = []
    #print('Loding training data...')
    for x, y in chord_training_generator:
        chord_training_data.append([x, y])
    chord_val_data = []
    #print()
    #print('Loading val data...')
    for x, y in chord_val_generator:
        chord_val_data.append([x, y])
        
    chord_training_steps = len(chord_training_data)
    chord_training_data = yield_data(chord_training_data)
    chord_val_steps = len(chord_val_data)
    chord_val_data = yield_data(chord_val_data)

    # Get polyphonic data
    #print()
    #print('Loading polyphonic data...')
    poly_training_generator = poly_data_generator(train_filenames, infinite=False, **poly_config)
    poly_val_generator = poly_data_generator(val_filenames, infinite=False, **poly_config)
    poly_training_data = []
    #print('Loading training data...')
    for x, y in poly_training_generator:
        poly_training_data.append([x, y])
    poly_val_data = []
    #print()
    #print('Loading val data...')
    for x, y in poly_val_generator:
        poly_val_data.append([x, y])

    poly_training_steps = len(poly_training_data)
    poly_training_data = yield_data(poly_training_data)
    poly_val_steps = len(poly_val_data)
    poly_val_data = yield_data(poly_val_data)

    chord_model.evaluate(chord_val_data, steps=chord_val_steps)
    poly_model.evaluate(poly_val_data, steps=poly_val_steps)
