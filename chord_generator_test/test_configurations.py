import os
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
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Constant
import load_data as load
from generator import chord_data_generator, poly_data_generator, count_steps, embed_poly_chords
import math
import time
import config as conf

class FullModel:
    def __init__(self, chord_config, poly_config):
        self.chord_config = chord_config
        self.poly_config = poly_config
        self.history = {
            'chord_train':{'loss':[], 'acc':[]}, 
            'chord_val':{'loss':[], 'acc':[]},
            'poly_train':{'loss':[], 'acc':[]},
            'poly_val':{'loss':[], 'acc':[]}
        }

    def _create_chord_model(self):
        batch_size = self.chord_config['batch_size']
        vocabulary = self.chord_config['vocabulary']
        max_steps = self.chord_config['max_steps']
        embedding_size = self.chord_config['embedding_size']
        chord_interval = self.chord_config['chord_interval']
        num_notes = self.chord_config['num_notes']
        lstm_size = self.chord_config['lstm_size']
        learning_rate = self.chord_config['learning_rate']

        optimizer = Adam(learning_rate=learning_rate)
        activation_function = 'softmax'
        loss = 'categorical_crossentropy'

        #Create chord embedding
        chord_input = Input(
            shape=(max_steps, 1), 
            batch_size=batch_size,
            name='chord_input'
        )
        embedding = Embedding(
            vocabulary,
            embedding_size, 
            name='chord_embedding'
        )(chord_input)

        #Create melody input
        melody_input = Input(
            shape=(max_steps, chord_interval, num_notes), 
            batch_size=batch_size,
            name='melody_input'
        )
        convLayer = Conv1D(
            num_notes, 
            chord_interval, 
            input_shape=(chord_interval, num_notes), 
            name='melody_conv'
        )(melody_input)

        #Concat chord and melody input
        lstm_data = concatenate([embedding, convLayer])
        lstm_data = Reshape((-1, num_notes+embedding_size))(lstm_data)


        #LSTM layer
        lstm = LSTM(
            lstm_size,
            stateful=True,
            return_sequences=True
        )(lstm_data)

        #Dense layer and activation
        dense = Dense(vocabulary, name='dense')(lstm)
        activation = Activation(activation_function)(dense)

        #Create model
        self.chord_model = Model(inputs=[chord_input, melody_input], outputs=activation)

        self.chord_model.compile(optimizer, loss, metrics=['accuracy'])
        self.chord_model.summary()

    def _create_poly_model(self):
        lstm_size = self.poly_config['lstm_size']
        batch_size = self.poly_config['batch_size']
        max_steps = self.poly_config['max_steps']
        num_notes = self.poly_config['num_notes']
        chord_interval = self.poly_config['chord_interval']
        embedding_size = self.poly_config['embedding_size']
        chord_vocab = self.chord_config['vocabulary']
        vocabulary = self.poly_config['vocabulary']
        learning_rate = self.poly_config['learning_rate']

        optimizer = Adam(learning_rate=learning_rate)
        activation_function = 'sigmoid'
        loss = 'binary_crossentropy'

        embedding_weights = self.chord_model.layers[2].get_weights()[0]
        
        #Create chord embedding
        chord_input = Input(
            shape=(max_steps, 1), 
            batch_size=batch_size,
            name='chord_input'
        )
        embedding = Embedding(
            chord_vocab, 
            embedding_size, 
            embeddings_initializer=Constant(embedding_weights), 
            trainable=False,
            name='chord_embedding'
        )(chord_input)
        embedding = Reshape((-1, embedding_size))(embedding)

        #Create melody and counter input
        x_input = Input(
            shape=(max_steps, num_notes+chord_interval),
            batch_size=batch_size,
            name='x_input'
        )

        #Concat chord and melody input
        lstm_data = concatenate([embedding, x_input])

        #LSTM layer
        lstm = LSTM(
            lstm_size, 
            input_shape=(max_steps, num_notes + chord_interval + embedding_size), 
            stateful=True, 
            return_sequences=True
        )(lstm_data)

        #Dense layer and activation
        dense = Dense(vocabulary)(lstm)
        activation = Activation(activation_function)(dense)

        #Create model
        self.poly_model = Model(inputs=[chord_input, x_input], outputs=activation)

        self.poly_model.compile(optimizer, loss, metrics=['accuracy'])
        self.poly_model.summary()

    def _test_chord_model(self, val_x, val_y):
        return 0, 0

    def _test_poly_model(self, val_x, val_y):
        return 0, 0

    def train_chord_model(self, train_x, train_y, val_x, val_y):
        batch_size = self.chord_config['batch_size']
        epochs = self.chord_config['epochs']

        self._create_chord_model()
        best_val_acc = 0
        best_epoch = 0
        best_model = None
        for e in range(1, epochs+1):
            print(f'Training epoch {e} of {epochs}...')
            epoch_start_time = time.time()
            loss = 0
            acc = 0
            progress = Progbar(len(train_x))
            for batch in range(len(train_x)):
                x = train_x[batch]
                y = train_y[batch]
                hist = self.chord_model.fit(x, y, batch_size=batch_size, epochs=1, shuffle=False, verbose=0)
                self.chord_model.reset_states()
                loss += hist.history['loss'][0]
                acc += hist.history['accuracy'][0]
                progress.update(batch+1)

            mean_loss = loss / len(train_x)
            mean_acc = acc / len(train_x)
            mean_val_loss, mean_val_acc = self._test_chord_model(val_x, val_y)
            self.history['chord_train']['loss'].append(mean_loss)
            self.history['chord_train']['acc'].append(mean_acc)
            self.history['chord_val']['loss'].append(mean_val_loss)
            self.history['chord_val']['acc'].append(mean_val_acc)

            if mean_val_acc > best_val_acc:
                best_epoch = e
                best_model = self.chord_model

            epoch_end_time = time.time()
            print(f'Finished epoch in {epoch_end_time - epoch_start_time:.2f} seconds.')
            print(f'loss: {mean_loss:.3f}, acc: {mean_acc:.3f}')
            print(f'val_loss: {mean_val_loss:.3f}, val_acc: {mean_val_acc:.3f}')

        print(f'Done training chord model for {epochs} epochs. Best epoch: {best_epoch}')
        self.best_chord_model = best_model
    
    def train_poly_model(self, train_x, train_y, val_x, val_y):
        batch_size = self.poly_config['batch_size']
        epochs = self.poly_config['epochs']

        self._create_poly_model()
        best_val_acc = 0
        best_epoch = 0
        best_model = None
        for e in range(1, epochs+1):
            print(f'Training epoch {e} of {epochs}...')
            epoch_start_time = time.time()
            loss = 0
            acc = 0
            progress = Progbar(len(train_x))
            for batch in range(len(train_x)):
                x = train_x[batch]
                y = train_y[batch]
                hist = self.poly_model.fit(x, y, batch_size=batch_size, epochs=1, shuffle=False, verbose=0)
                self.poly_model.reset_states()
                loss += hist.history['loss'][0]
                acc += hist.history['accuracy'][0]
                progress.update(batch)

            mean_loss = loss / len(train_x)
            mean_acc = acc / len(train_x)
            mean_val_loss, mean_val_acc = self._test_poly_model(val_x, val_y)
            self.history['poly_train']['loss'].append(mean_loss)
            self.history['poly_train']['acc'].append(mean_acc)
            self.history['poly_val']['loss'].append(mean_val_loss)
            self.history['poly_val']['acc'].append(mean_val_acc)

            if mean_val_acc > best_val_acc:
                best_epoch = e
                best_model = self.poly_model

            epoch_end_time = time.time()
            print(f'Finished epoch steps in {epoch_end_time - epoch_start_time:.2f} seconds.')
            print(f'loss: {mean_loss:.3f}, acc: {mean_acc:.3f}')
            print(f'val_loss: {mean_val_loss:.3f}, val_acc: {mean_val_acc:.3f}')

        print(f'Done training poly model for {epochs} epochs. Best epoch: {best_epoch}')
        self.best_poly_model = best_model

if __name__ == "__main__":
    chord_config = {'lstm_size':512,
                    'batch_size':128,
                    'val_batch_size':128,
                    'learning_rate':0.00001,
                    'embedding_size':10,
                    'vocabulary':100,
                    'max_steps':8,
                    'num_notes':60,
                    'chord_interval':16,
                    'epochs':100}

    poly_config = {'lstm_size':512,
                   'batch_size':128,
                   'val_batch_size':128,
                   'learning_rate':0.0001,
                   'embedding_size':10,
                   'vocabulary':(conf.num_notes*3)+24,
                   'max_steps':128,
                   'num_notes':60,
                   'chord_interval':16,
                   'epochs':100}
    model = FullModel(chord_config, poly_config)

    train_filenames, val_filenames = load.get_trainval_filenames(rand_data=conf.random_data)
    
    # Get chord data
    print('Loading chord data...')
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
        
    # Get polyphonic data
    print('Loading polyphonic data...')
    poly_training_generator = poly_data_generator(train_filenames, infinite=False, **poly_config)
    poly_val_generator = poly_data_generator(val_filenames, infinite=False, **poly_config)
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

    print('Training chord model...')
    model.train_chord_model(chord_training_x, chord_training_y, chord_val_x, chord_val_y)
    print('Training polyphonic model...')
    model.train_poly_model(poly_training_x, poly_training_y, poly_val_x, poly_val_y)
