import os
from queue import Full
import numpy as np
import matplotlib.pyplot as plt
from six import python_2_unicode_compatible
import tensorflow as tf
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
import math
import time
import datetime
import config as conf
import _pickle as pickle

np.random.seed(2020)

chord_input = Input(
    shape=(128, 1), 
    name='chord_input'
)
embedding = Embedding(
    100, 
    10, 
    trainable=False,
    name='chord_embedding'
)(chord_input)
embedding = Reshape((-1, 10))(embedding)

#Create melody and counter input
x_input = Input(
    shape=(128, 60+16),
    name='x_input'
)

#Concat chord and melody input
lstm_data = concatenate([embedding, x_input])

#LSTM layer
lstm = LSTM(
    512, 
    input_shape=(128, 60 + 16 + 10), 
    return_sequences=True
)(lstm_data)

#Dense layer and activation
dense = Dense(100)(lstm)
activation = Activation('sigmoid')(dense)

#Create model
poly_model = Model(inputs=[chord_input, x_input], outputs=activation)
poly_model.call = tf.function(poly_model.call)
times = []
for i in range(100):
    start_time = time.process_time()
    pred = poly_model([np.random.rand(1, 128), np.random.rand(1, 128, 76)], training=False)
    times.append(time.process_time() - start_time)
print('Mean time: ', np.mean(times))