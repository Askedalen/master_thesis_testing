import os
from pickle import INST
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from tensorflow.keras.layers import LSTM
from numpy.testing._private.utils import IS_PYPY
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import numpy as np
import matplotlib.pyplot as plt
import automatic_music_accompaniment_master.code.utils as utils
from keras_batch_generator import KerasBatchGenerator
import parse_midi as parse

NUMBER_FEATURES = 128#utils.NUMBER_FEATURES
INSTRUMENTS = 5

experiment_path = os.path.join('cheriell_model', '')
num_steps = 32
batch_size = 256
skip_step = 3
lr = 0.0001
vocabulary = NUMBER_FEATURES*INSTRUMENTS
hidden_size = 1024
num_epochs = 5

train_data, valid_data = parse.load_datafiles()

train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary, skip_step=skip_step)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary, skip_step=skip_step)

# Model definition
model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(num_steps, NUMBER_FEATURES)))
model.add(LSTM(512, return_sequences=True))
model.add(TimeDistributed(Dense(vocabulary, activation='sigmoid')))
optimizer = Adam(learning_rate=lr)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

checkpointer = ModelCheckpoint(filepath=experiment_path + 'model{epoch:02d}.hdf5', verbose=1)

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.logs = []
        
    def on_epoch_end(self, epoch, logs={}):
        # plot graph on_epoch_end
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        plt.figure()
        plt.plot(self.x, self.losses, 'r--', label='loss')
        plt.plot(self.x, self.val_losses, 'g--', label='val_loss')
        plt.title('loss vs. valid_loss on epoch{}'.format(self.i))
        plt.legend()
        plt.grid()
        plt.savefig(experiment_path + 'loss vs.valid_loss epoch{}.svg'.format(self.i), format='svg')

plot_losses = PlotLosses()

steps_per_epoch = 0
for i in range(len(train_data)):
    steps_per_epoch += len(train_data[i][0] - num_steps) // (batch_size * skip_step)
validation_steps = 0
for i in range(len(valid_data)):
    validation_steps += len(valid_data[i][0] - num_steps) // (batch_size * skip_step)


His = model.fit_generator(train_data_generator.generate_test(), steps_per_epoch, validation_data=valid_data_generator.generate_test(), validation_steps=validation_steps, callbacks=[checkpointer, plot_losses])

# plot losses in the end
plt.figure()
plt.plot(His.history['loss'], 'r--', label='loss')
plt.plot(His.history['val_loss'], 'g--', label='validation loss')
plt.title('final loss vs. valid_loss')
plt.legend()
plt.grid(linestyle = "--")
plt.savefig(experiment_path + 'final loss vs. valid_loss.svg', format='svg')

# plot accuracies in the end
plt.figure()
plt.plot(His.history['accuracy'], 'r--', label='accuracy')
plt.plot(His.history['val_accuracy'], 'g--', label='validation accuracy')
plt.title('final accuracy vs. valid_accuracy')
plt.legend()
plt.grid(linestyle = "--")
plt.savefig(experiment_path + 'final accuracy vs. valid_accuracy.svg', format='svg')

# save losses and accuracies during training
np.save(experiment_path + 'History.loss.npy', His.history['loss'])
np.save(experiment_path + 'History.val_loss.npy', His.history['val_loss'])
np.save(experiment_path + 'History.acc.npy', His.history['accuracy'])
np.save(experiment_path + 'History.val_acc.npy', His.history['val_accuracy'])

# save the final model
model.save(experiment_path + 'final_model.hdf5')