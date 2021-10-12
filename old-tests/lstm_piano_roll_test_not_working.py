import numpy as np
import tensorflow as tf
import pretty_midi as pm
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Activation
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import time
from tensorflow.python.keras.callbacks import ModelCheckpoint
import parse_MIDI_to_piano_roll_test as parse
import math

additional_metrics = ['accuracy']
batch_size = 64
hidden_size = 500
num_steps = 30
vocabulary = 128
loss_function = BinaryCrossentropy()
number_of_epochs = 10
optimizer = Adam()
validation_split = 0.20
verbosity_mode = 1
use_dropout = False

data = parse.getSongs(2)

data_x = data[0].T
data_x[data_x > 0] = 1 
data_t = data[1].T
data_t[data_t > 0] = 1

#data_x = data_x.reshape((1, data_x.shape[0], data_x.shape[1]))
#data_t = data_t.reshape((1, data_t.shape[0], data_t.shape[1]))

test1 = np.zeros((math.floor(data_x.shape[0] / 50), 50, 128))
test2 = np.zeros((math.floor(data_t.shape[0] / 50), 50, 128))
for i in range(math.floor(data_x.shape[0] / (data_x.shape[0] / 50))):
    test1[i, :, :] = data_x[i*50:(i+1)*50, :]
    test2[i, :, :] = data_t[i*50:(i+1)*50, :]

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(50, 128)))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
""" model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3)) """
model.add(Dense(vocabulary)) 
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

filepath="./Checkpoints/checkpoint_model_{epoch:02d}.hdf5"
model_save_callback = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='auto', period=5)

model.fit(test1, test2, epochs=5, verbose=1, batch_size=128, callbacks=[model_save_callback])

model.save('lstm_test_0_model.md5')
