from calendar import c
from http.client import REQUESTED_RANGE_NOT_SATISFIABLE
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
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.python.keras.backend import reshape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.backend import reshape, function
from tensorflow.python.keras.layers.core import Flatten, Reshape
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.initializers import Constant
from tensorflow.random import set_seed
import load_data as load
from generator import baseline_data_generator, chord_data_generator, poly_data_generator, count_steps, embed_poly_chords
import math
import time
import datetime
import config as conf
import _pickle as pickle

class BaselineModel:
    def __init__(self, config):
        self.config = config
        self.model_name = f"baseline_d{datetime.datetime.now().strftime('%y%m%d_t%H%M')}"
        self.history = {
            'train':{'loss':[], 'acc':[]},
            'val':{'loss':[], 'acc':[]}
        }

    def _create_model(self):
        batch_size = self.config['batch_size']
        max_steps = self.config['max_steps']
        num_notes = self.config['num_notes']
        counter_size = self.config['counter_size']
        lstm_size = self.config['lstm_size']
        learning_rate = self.config['learning_rate']
        output_size = self.config['output_size']

        optimizer = Adam(learning_rate=learning_rate)
        activation_function = 'sigmoid'
        loss = 'binary_crossentropy'

        self.model = Sequential()
        self.model.add(LSTM(lstm_size, return_sequences=True, input_shape=(max_steps, num_notes+counter_size)))
        self.model.add(Dense(output_size))
        self.model.add(Activation(activation_function))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(self.model.summary())

    def train_model(self, train_data, train_steps, val_data, val_steps):
        epochs = self.config['epochs']
        verbose = self.config['verbose']

        self._create_model()
        best_val_acc = 0
        best_epoch = 0
        best_weights = self.model.get_weights()

        start_time = time.process_time()
        for e in range(1, epochs+1):
            print(f'Training epoch {e} of {epochs}')
            hist = self.model.fit(
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

            self.history['train']['loss'].append(loss)
            self.history['train']['acc'].append(acc)
            self.history['val']['loss'].append(val_loss)
            self.history['val']['acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e
                best_weights = self.model.get_weights()
        
        end_time = time.process_time()
        self.time = end_time - start_time
        self.best_val_acc = best_val_acc
        self.best_epoch = best_epoch
        self.best_weights = best_weights

    def plot_history(self):
        self.test_dir = os.path.join(conf.results_dir, 'tests', self.model_name)
        if not os.path.exists(os.path.join(self.test_dir, 'models')):
            os.makedirs(os.path.join(self.test_dir, 'models'))

        loss_roof = math.ceil(np.max([self.history['train']['loss'], self.history['val']['loss']]))

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(self.history['train']['loss'], 'r--', label='Loss')
        axs[0].plot(self.history['val']['loss'], 'g--', label='Validation loss')
        axs[0].set_title('Training Loss vs Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlim(0, self.config['epochs']-1)
        axs[0].set_ylim(0, loss_roof)
        axs[0].legend()

        axs[1].plot(self.history['train']['acc'], 'r--', label='Accuracy')
        axs[1].plot(self.history['val']['acc'], 'g--', label='Validation accuracy')
        axs[1].set_title('Training Accuracy vs. Validation Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].set_xlim(0, self.config['epochs']-1)
        axs[1].set_ylim(0, 1)
        axs[1].legend()
        fig.suptitle('Training history')
        fig.savefig(os.path.join(self.test_dir, 'history.png'))

    def save_model(self):
        self.model.set_weights(self.best_weights)
        self.model.save(os.path.join(self.test_dir, 'model.pb'))

def yield_data(data):
    while True:
        for x, y in data:
            yield x, y

if __name__ == "__main__":
    accs = []
    epochs = []
    times = []
    train_filenames, val_filenames = load.get_trainval_filenames()
    train_filenames = train_filenames[:12000]
    val_filenames = val_filenames[:3000]
    print('Loading data...')
    training_generator = baseline_data_generator(train_filenames)
    val_generator = baseline_data_generator(val_filenames)
    training_data = []
    for x, y in training_generator:
        training_data.append([x, y])
    val_data = []
    for x, y in val_generator:
        val_data.append([x, y])

    config = {
        'batch_size':128,
        'max_steps':128,
        'num_notes':60,
        'counter_size':16,
        'lstm_size':1024,
        'learning_rate':0.0001,
        'output_size':204,
        'epochs':100,
        'verbose':0
    }

    training_steps = len(training_data)
    training_data = yield_data(training_data)
    val_steps = len(val_data)
    val_data = yield_data(val_data)

    for i in range(1):
        baseline_model = BaselineModel(config)
        baseline_model.train_model(training_data, training_steps, val_data, val_steps)
        baseline_model.plot_history()
        baseline_model.save_model()
        accs.append(baseline_model.best_val_acc)
        epochs.append(baseline_model.best_epoch)
        times.append(baseline_model.time)
    print('Accuracies: ', accs)
    print('Best epochs:', epochs)
    print('Times:', times)

     
