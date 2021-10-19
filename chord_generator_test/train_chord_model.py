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
import load_data
from generator import chord_data_generator
import math
import time
from config import *

TESTING = True

lstm_size = 512
learning_rate = 0.00001
num_notes = 128
embedding_size = 10
vocabulary = 100

if TESTING:
    batch_size = 8
    val_batch_size = 4
    epochs = 2
    num_songs = 40
    verbose = 1
else:
    batch_size = 32
    val_batch_size = 16
    epochs = 100
    num_songs = 0
    verbose = 2

params = {'max_steps':1000,
          'num_notes':num_notes,
          'vocabulary':vocabulary}

train_filenames, val_filenames = get_trainval_filenames(num_songs)
training_generator = chord_data_generator(train_filenames, batch_size=batch_size, **params)
val_generator = chord_data_generator(val_filenames, batch_size=val_batch_size, **params)
optimizer = Adam(learning_rate=learning_rate)
loss = 'categorical_crossentropy'
print('Creating model...')

#Create chord embedding
chord_input = Input(shape=(1,))
embedding = Embedding(vocabulary, embedding_size)(chord_input)

#Create melody input
melody_input = Input(shape=(1,num_notes,))

#Concat chord and melody input
lstm_data = concatenate([embedding, melody_input])
lstm_data = Reshape((-1,num_notes+embedding_size))(lstm_data)

#LSTM layer
lstm = LSTM(lstm_size, return_sequences=True)(lstm_data)

#Dense layer and activation
time_dist = TimeDistributed(Dense(vocabulary))(lstm)
activation = Activation('softmax')(time_dist)

#Create model
model = Model(inputs=[chord_input, melody_input], outputs=activation)

model.compile(optimizer, loss, metrics=['accuracy'])
model.summary()

losses = np.zeros((2, epochs))
accuracies = np.zeros((2, epochs))

def train():
    steps_per_epoch = math.floor(len(train_filenames) / batch_size)
    validation_steps = math.floor(len(val_filenames) / val_batch_size)
    print('Training...')
    start_time = time.time()
    for e in range(1, epochs+1):
        print('Epoch', e, 'of', epochs)
        hist = model.fit(training_generator, validation_data=val_generator, epochs=1, shuffle=False, verbose=verbose, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
        model.reset_states()

        losses[0, e-1] = hist.history['loss'][0]
        losses[1, e-1] = hist.history['val_loss'][0]
        accuracies[0, e-1] = hist.history['accuracy'][0]
        accuracies[1, e-1] = hist.history['val_accuracy'][0]
        model.save(os.path.join(results_dir, 'models', f'epoch{e:03d}.hdf5'))
    end_time = time.time()
    print(f'Finished training in {end_time - start_time} seconds')

def plot_results():        
    plt.figure()
    plt.plot(losses[0], 'r--', label='Loss')
    plt.plot(losses[1], 'g--', label='Validation loss')
    plt.title('Training Loss vs Validtation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_vs_val_loss.png'))

    plt.figure()
    plt.plot(accuracies[0], 'r--', label='Accuracy')
    plt.plot(accuracies[1], 'g--', label='Validation accuracy')
    plt.title('Training accuracy vs. Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'acc_vs_val_acc.png'))

def print_results():
    print()
    print(f'Finished training {epochs} epochs.')
    print('Results:')
    print(f'Mean loss: {np.mean(losses[0])}')
    print(f'Mean val_loss: {np.mean(losses[1])}')
    print(f'Mean accuracy: {np.mean(accuracies[0])}')
    print(f'Mean val_accuracy: {np.mean(accuracies[1])}')
    print()
    print(f'Best loss: {np.min(losses[0])}')
    print(f'Best val_loss: {np.min(losses[1])}')
    print(f'Best accuracy: {np.max(accuracies[0])}')
    print(f'Best val_accuracy: {np.max(accuracies[1])}')
    print()
    print(f'Best epoch: {np.argmax(accuracies[1]) + 1}')

train()
plot_results()
print_results()
