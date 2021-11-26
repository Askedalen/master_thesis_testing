import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import Adam
from load_data import get_trainval_filenames
from generator import poly_data_generator, count_steps
import load_data
import math
import time
import config as conf

model_path = os.path.join(conf.chord_model_dir, 'epoch099.hdf5')

lstm_size = 512
learning_rate = 0.00001
num_notes = conf.num_notes
embedding_size = 10
vocabulary = (num_notes*3) + 24
max_steps = 128
chord_interval = 16
load_all_data = True

if conf.testing:
    batch_size = 128
    val_batch_size = 128
    epochs = 2
    num_songs = 0
    verbose = 1
else:
    batch_size = 128
    val_batch_size = 128
    epochs = 100
    num_songs = 0
    verbose = 2

params = {'max_steps':max_steps,
          'chord_interval':chord_interval,
          'num_notes':num_notes,
          'rand_data':conf.random_data}

train_filenames, val_filenames = get_trainval_filenames(num_songs, rand_data=conf.random_data)
chord_embedding = load_data.ChordEmbedding(model_path)
print(f'Counting steps for {len(train_filenames) + len(val_filenames)} files')
training_steps = count_steps(train_filenames, batch_size, generator_num=1, chord_embedding=chord_embedding, **params)
val_steps = count_steps(val_filenames, val_batch_size, generator_num=1, chord_embedding=chord_embedding, **params)
print(f'Training steps: {training_steps} \r\nVal steps: {val_steps}')
if load_all_data:
    print('Loading all data...')
    training_generator = poly_data_generator(train_filenames, chord_embedding, batch_size=batch_size, infinite=False, **params)
    val_generator = poly_data_generator(val_filenames, chord_embedding, batch_size=val_batch_size, infinite=False, **params)

    training_x = []
    training_y = []
    for x, y in training_generator:
        training_x.append(x)
        training_y.append(y)

    val_x = []
    val_y = []
    for x, y in val_generator:
        val_x.append(x)
        val_y.append(y)
    print('Done loading data')
else:
    training_generator = poly_data_generator(train_filenames, chord_embedding, batch_size=batch_size, **params)
    val_generator = poly_data_generator(val_filenames, chord_embedding, batch_size=val_batch_size, **params)

optimizer = Adam(learning_rate=learning_rate)
loss = 'binary_crossentropy'
print('Creating model...')

model = Sequential()
model.add(LSTM(lstm_size, batch_size=batch_size, input_shape=(max_steps, num_notes + chord_interval + embedding_size), stateful=True, return_sequences=True))
model.add(Dense(vocabulary))
model.add(Activation('sigmoid'))

model.compile(optimizer, loss, metrics=['accuracy'])
model.summary()

losses = np.zeros((2, epochs))
accuracies = np.zeros((2, epochs))

def train():
    print('Training...')
    start_time = time.time()
    for e in range(1, epochs+1):
        print('Epoch', e, 'of', epochs)
        epoch_start_time = time.time()
        if load_all_data:
            loss = 0
            acc = 0
            for batch in range(len(training_x)):
                x = training_x[batch]
                y = training_y[batch]
                hist = model.fit(x, y, batch_size=batch_size, epochs=1, shuffle=False, verbose=0)
                model.reset_states()
                loss += hist.history['loss'][0]
                acc += hist.history['accuracy'][0]

            mean_loss = loss / len(training_x)
            mean_acc = acc / len(training_x)
            losses[0, e-1] = mean_loss
            accuracies[0, e-1] = mean_acc
            print(f'Finished {training_steps} steps - loss: {mean_loss}, acc: {mean_acc}')
            test(e)
        else:
            hist = model.fit(training_generator, validation_data=val_generator, epochs=1, shuffle=False, verbose=verbose, steps_per_epoch=training_steps, validation_steps=val_steps)
            model.reset_states()
            losses[0, e-1] = hist.history['loss'][0]
            losses[1, e-1] = hist.history['val_loss'][0]
            accuracies[0, e-1] = hist.history['accuracy'][0]
            accuracies[1, e-1] = hist.history['val_accuracy'][0]
        
        epoch_end_time = time.time()
        print(f'Epoch time: {epoch_end_time - epoch_start_time}s')
        model.save(os.path.join(conf.poly_model_dir, f'epoch{e:03d}.hdf5'))
    end_time = time.time()
    print(f'Finished training in {end_time - start_time} seconds')

def test(e):
    val_loss = 0
    val_acc = 0
    for batch in range(len(val_x)):
        x = val_x[batch]
        y = val_y[batch]
        hist = model.evaluate(x, y, verbose=0)
        val_loss += hist[0]
        val_acc += hist[1]
    mean_val_loss = val_loss / len(val_x)
    mean_val_acc = val_acc / len(val_x)
    losses[1, e-1] = mean_val_loss
    accuracies[1, e-1] = mean_val_acc
    print(f'val_loss:{mean_val_loss}, val_acc:{mean_val_acc}')

def plot_results():        
    plt.figure()
    plt.plot(losses[0], 'r--', label='Loss')
    plt.plot(losses[1], 'g--', label='Validation loss')
    plt.title('Training Loss vs Validtation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(conf.results_dir, 'poly_loss_vs_val_loss.png'))

    plt.figure()
    plt.plot(accuracies[0], 'r--', label='Accuracy')
    plt.plot(accuracies[1], 'g--', label='Validation accuracy')
    plt.title('Training accuracy vs. Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(conf.results_dir, 'poly_acc_vs_val_acc.png'))

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
