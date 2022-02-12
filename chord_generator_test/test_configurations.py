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
import _pickle as pickle

np.random.seed(42)
set_seed(42)

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
        self.model_name = f"d{datetime.datetime.now().strftime('%y%m%d_t%H%M')}" 
                          
        self.test_dir = os.path.join(conf.results_dir, 'tests', self.model_name)
        if not os.path.exists(os.path.join(self.test_dir, 'models')):
            os.makedirs(os.path.join(self.test_dir, 'models'))

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
            return_sequences=True
        )(lstm_data)

        #Dense layer and activation
        dense = Dense(vocabulary, name='dense')(lstm)
        activation = Activation(activation_function)(dense)

        #Create model
        self.chord_model = Model(inputs=[chord_input, melody_input], outputs=activation)

        self.chord_model.compile(optimizer, loss, metrics=['accuracy'])
        #self.chord_model.summary()

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

        embedding_weights = self.best_chord_weights[0]

        #Create chord embedding
        chord_input = Input(
            shape=(max_steps, 1), 
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
            name='x_input'
        )

        #Concat chord and melody input
        lstm_data = concatenate([embedding, x_input])

        #LSTM layer
        lstm = LSTM(
            lstm_size, 
            input_shape=(max_steps, num_notes + chord_interval + embedding_size), 
            return_sequences=True
        )(lstm_data)

        #Dense layer and activation
        dense = Dense(vocabulary)(lstm)
        activation = Activation(activation_function)(dense)

        #Create model
        self.poly_model = Model(inputs=[chord_input, x_input], outputs=activation)

        self.poly_model.compile(optimizer, loss, metrics=['accuracy'])
        #self.poly_model.summary()

    def train_chord_model(self, train_data, train_steps, val_data, val_steps):
        epochs = self.chord_config['epochs']
        verbose = self.chord_config['verbose']
        self._create_chord_model()

        best_val_acc = 0
        best_epoch = 0
        best_weights = self.chord_model.get_weights()
        start_time = time.time()
        for e in range(1, epochs+1):
            #print(f'Training epoch {e} of {epochs}...')
            hist = self.chord_model.fit(
                train_data, 
                validation_data=val_data, 
                steps_per_epoch=train_steps, 
                validation_steps=val_steps, 
                epochs=1, 
                shuffle=False, 
                verbose=verbose
            )

            loss = hist.history['loss'][0]
            acc = hist.history['accuracy'][0]
            val_loss = hist.history['val_loss'][0]
            val_acc = hist.history['val_accuracy'][0]
            self.history['chord_train']['loss'].append(loss)
            self.history['chord_train']['acc'].append(acc)
            self.history['chord_val']['loss'].append(val_loss)
            self.history['chord_val']['acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e
                best_weights = self.chord_model.get_weights()

        end_time = time.time()
        print(f'Trained chord model for {epochs} epochs in {end_time - start_time:.2f} seconds. Best epoch: {best_epoch}')
        print(f"mean loss:{np.mean(self.history['chord_train']['loss'])}, best acc:{np.max(self.history['chord_train']['acc'])}")
        print(f"mean val loss:{np.mean(self.history['chord_val']['loss'])}, best val acc:{np.max(self.history['chord_val']['acc'])}")
        print()
        self.best_chord_weights = best_weights
    
    def train_poly_model(self, train_data, train_steps, val_data, val_steps):
        epochs = self.poly_config['epochs']
        verbose = self.poly_config['verbose']

        self._create_poly_model()
        best_val_acc = 0
        best_epoch = 0
        best_weights = self.poly_model.get_weights()

        start_time = time.time()
        for e in range(1, epochs+1):
            #print(f'Training epoch {e} of {epochs}...')
            hist = self.poly_model.fit(
                train_data, 
                validation_data=val_data, 
                steps_per_epoch=train_steps, 
                validation_steps=val_steps, 
                epochs=1, 
                shuffle=False, 
                verbose=verbose
            )

            loss = hist.history['loss'][0]
            acc = hist.history['accuracy'][0]
            val_loss = hist.history['val_loss'][0]
            val_acc = hist.history['val_accuracy'][0]
            self.history['poly_train']['loss'].append(loss)
            self.history['poly_train']['acc'].append(acc)
            self.history['poly_val']['loss'].append(val_loss)
            self.history['poly_val']['acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e
                best_weights = self.poly_model.get_weights()

        end_time = time.time()
        print(f'Trained polyphonic model for {epochs} epochs in {end_time - start_time:.2f} seconds. Best epoch: {best_epoch}')
        print(f"mean loss:{np.mean(self.history['poly_train']['loss'])}, best acc:{np.max(self.history['poly_train']['acc'])}")
        print(f"mean val loss:{np.mean(self.history['poly_val']['loss'])}, best val acc:{np.max(self.history['poly_val']['acc'])}")
        print()
        self.best_poly_weights = best_weights
        self.best_val_acc = best_val_acc

    def plot_history(self):
        chord_loss_roof = math.ceil(np.max([self.history['chord_train']['loss'], self.history['chord_val']['loss']]))
        poly_loss_roof = math.ceil(np.max([self.history['poly_train']['loss'], self.history['poly_val']['loss']]))

        c_fig, c_axs = plt.subplots(1, 2, figsize=(10, 5))
        c_axs[0].plot(self.history['chord_train']['loss'], 'r--', label='Loss')
        c_axs[0].plot(self.history['chord_val']['loss'], 'g--', label='Validation loss')
        c_axs[0].set_title('Training Loss vs Validtation Loss')
        c_axs[0].set_xlabel('Epoch')
        c_axs[0].set_ylabel('Loss')
        c_axs[0].set_xlim(0, self.chord_config['epochs']-1)
        c_axs[0].set_ylim(0, chord_loss_roof)
        c_axs[0].legend()

        c_axs[1].plot(self.history['chord_train']['acc'], 'r--', label='Accuracy')
        c_axs[1].plot(self.history['chord_val']['acc'], 'g--', label='Validation accuracy')
        c_axs[1].set_title('Training accuracy vs. Validation Accuracy')
        c_axs[1].set_xlabel('Epoch')
        c_axs[1].set_ylabel('Accuracy (%)')
        c_axs[1].set_xlim(0, self.chord_config['epochs']-1)
        c_axs[1].set_ylim(0, 1)
        c_axs[1].legend()
        c_fig.suptitle('Chord training history')
        c_fig.savefig(os.path.join(self.test_dir, 'chord_history.png'))

        p_fig, p_axs = plt.subplots(1, 2, figsize=(10, 5))
        p_axs[0].plot(self.history['poly_train']['loss'], 'r--', label='Loss')
        p_axs[0].plot(self.history['poly_val']['loss'], 'g--', label='Validation loss')
        p_axs[0].set_title('Training Loss vs Validtation Loss')
        p_axs[0].set_xlabel('Epoch')
        p_axs[0].set_ylabel('Loss')
        p_axs[0].set_xlim(0, self.poly_config['epochs']-1)
        p_axs[0].set_ylim(0, poly_loss_roof)
        p_axs[0].legend()

        p_axs[1].plot(self.history['poly_train']['acc'], 'r--', label='Accuracy')
        p_axs[1].plot(self.history['poly_val']['acc'], 'g--', label='Validation accuracy')
        p_axs[1].set_title('Training accuracy vs. Validation Accuracy')
        p_axs[1].set_xlabel('Epoch')
        p_axs[1].set_ylabel('Accuracy (%)')
        p_axs[1].set_xlim(0, self.poly_config['epochs']-1)
        p_axs[1].set_ylim(0, 1)
        p_axs[1].legend()
        p_fig.suptitle('Polyphonic training history')
        p_fig.savefig(os.path.join(self.test_dir, 'poly_history.png'))

    def save_models(self):
        print('Saving chord model')
        self.chord_model.set_weights(self.best_chord_weights)
        self.chord_model.save(os.path.join(self.test_dir, 'models', 'chord_model.pb'))
        
        print('Saving poly model')
        self.poly_model.set_weights(self.best_poly_weights)
        self.poly_model.save(os.path.join(self.test_dir, 'models', 'poly_model.pb'))
        
        config_file = open(os.path.join(self.test_dir, "configs.txt"), "w")
        config_file.write('Chord config:\r\n')
        config_file.write(str(self.chord_config))
        config_file.write('\r\nPoly config:\r\n')
        config_file.write(str(self.poly_config))
        config_file.close()

    def get_best_val_acc(self):
        return self.best_val_acc

    def evaluate_saved(self, chord_val_data, chord_val_steps, poly_val_data, poly_val_steps):
        print('Testing stored chord model')
        test_chord_model = load_model(os.path.join(self.test_dir, 'models', 'chord_model.pb'))
        test_chord_model.compile(Adam(learning_rate=self.chord_config['learning_rate']), 'categorical_crossentropy', metrics=['accuracy'])
        
        print('Testing stored poly model')
        test_poly_model = load_model(os.path.join(self.test_dir, 'models', 'poly_model.pb'))
        test_poly_model.compile(Adam(learning_rate=self.poly_config['learning_rate']), 'binary_crossentropy', metrics=['accuracy'])
        

        self.chord_model.evaluate(chord_val_data, steps=chord_val_steps)
        self.poly_model.evaluate(poly_val_data, steps=poly_val_steps)
        

def yield_data(data):
    while True:
        for [x1, x2], y in data:
            yield [x1, x2], y

if __name__ == "__main__":
    ### TEST MODEL CREATION
    """ chord_config = {'batch_size':128,
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

    test = FullModel(chord_config, poly_config)
    test._create_chord_model()
    test._create_poly_model() """


    train_filenames, val_filenames = load.get_trainval_filenames(rand_data=conf.random_data)
    #train_filenames = train_filenames[:50]
    #val_filenames = val_filenames[:50]
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

    chord_config.update({'lstm_size':512,
                        'learning_rate':0.00001,
                        'embedding_size':10,
                        'epochs':100,
                        'verbose':0})

    poly_config.update({'lstm_size':1024,
                        'learning_rate':0.01,
                        'embedding_size':10,
                        'epochs':100,
                        'verbose':0})
    
    model = FullModel(chord_config, poly_config)

    print()
    print('Training chord model...')
    model.train_chord_model(chord_training_data, chord_training_steps, chord_val_data, chord_val_steps)
    print()
    print('Training polyphonic model...')
    model.train_poly_model(poly_training_data, poly_training_steps, poly_val_data, poly_val_steps)
    model.plot_history()
    model.save_models()
    model.evaluate_saved(chord_val_data, chord_val_steps, poly_val_data, poly_val_steps)
    print('Done training and saving models')
