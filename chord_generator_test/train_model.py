import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Activation, Embedding, Input, TimeDistributed
from tensorflow.keras.layers import Concatenate, add, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import reshape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.core import Reshape
from load_data import get_trainval_filenames
from generator import ChordMelodyGenerator, yield_generator_test
import load_data
import generator
import math

lstm_size = 512
batch_size = 32
val_batch_size = 32
learning_rate = 0.00001
epochs = 100

params = {'max_steps':1000,
          'num_notes':128,
          'vocabulary':100}


print('Loading data...')
#train_data, test_data = load_data.get_trainval_chords_and_melody(num_songs=10)
train_filenames, val_filenames = get_trainval_filenames()
training_generator = yield_generator_test(train_filenames, batch_size=batch_size, **params)
val_generator = yield_generator_test(val_filenames, batch_size=val_batch_size, **params)
optimizer = Adam(learning_rate=learning_rate)
loss = 'categorical_crossentropy'
print('Creating model...')

#Create chord embedding
chord_input = Input(shape=(1,))
embedding = Embedding(100, 100, name='test')(chord_input)

#Create melody input
melody_input = Input(shape=(1,128,))

#Concat chord and melody input
lstm_data = concatenate([embedding, melody_input], name='faen')
lstm_data = Reshape((1000,228))(lstm_data)

#LSTM layer
lstm = LSTM(lstm_size, return_sequences=True)(lstm_data)

#Dense layer and activation
time_dist = TimeDistributed(Dense(100))(lstm)
activation = Activation('softmax')(time_dist)

#Create model
model = Model(inputs=[chord_input, melody_input], outputs=activation)

model.compile(optimizer, loss)
model.summary()


def train():
    steps_per_epoch = math.floor(len(train_filenames) / batch_size)
    validation_steps = math.floor(len(val_filenames) / val_batch_size)
    print('Training...')
    total_train_loss = 0
    for e in range(1, epochs+1):
        print('Epoch', e, 'of', epochs)
        hist = model.fit(training_generator, validation_data=val_generator, epochs=1, shuffle=False, verbose=True, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
        model.reset_states()

    model.save('test.hdf5')
train()
