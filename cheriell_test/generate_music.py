from keras.models import load_model
import numpy as np
import automatic_music_accompaniment_master.code.utils as utils
import argparse
import parse_midi as parse
import pretty_midi as pm
import math

NUMBER_FEATURES = 128#utils.NUMBER_FEATURES
INSTRUMENTS = 5
num_steps = 32
vocabulary = NUMBER_FEATURES*INSTRUMENTS

def generate_midi(data, filename):
    number_of_ts = len(data[0])

    music = pm.PrettyMIDI()

    for i in range(INSTRUMENTS):
        piano_program = 0
        piano = pm.Instrument(program=0)

        for t in range(number_of_ts):
            for pitch in range(128):
                if data[i, t, pitch]:
                    start_time = utils.t_to_time(music, t)
                    end_t = t + 1
                    while end_t < number_of_ts and data[i, end_t, NUMBER_FEATURES - 3]:
                        end_t += 1
                    end_time = utils.t_to_time(music, end_t)
                    note = pm.Note(velocity=80, pitch=pitch, start=start_time, end=end_time)
                    piano.notes.append(note)
        music.instruments.append(piano)
    music.write(filename)

""" parser = argparse.ArgumentParser()
parser.add_argument('midi_file', type=str, default=None, help="The midi file for melody input")
parser.add_argument('--model_file', type=str, default='cheriell_model\\final_model.mdf5',  help='model file for accompaniment generation')
parser.add_argument('--diversity', type=float, default=0.6, help='diversity in accompaniment generation')
args = parser.parse_args() """

midi_data = parse.load_datafiles()[0] #args.midi_file
model_file = 'cheriell_model\\model20.hdf5' #args.model_file
diversity = 0.6 #args.diversity

model = load_model(model_file)
print("Loaded model")
while True:
    test_data = midi_data[np.random.randint(0, len(midi_data))]
    if test_data[3].max(): 
        num_notes = np.count_nonzero(np.any(test_data[3], axis=1))
        target_num_notes = test_data[3].shape[0] * 0.5
        if num_notes > (target_num_notes):
            break

# sample a note from the probability distribution.
# This helper function is copied from keras lstm examples at:
# https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(prediction, diversity=0.6):
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / diversity
    prediction_exp = np.exp(prediction)
    prediction = prediction_exp / np.sum(prediction_exp)
    probs = np.random.multinomial(1, prediction, 1)
    return np.argmax(probs)

#generate accompaniment
test_data[0, num_steps:, :] = 0
test_data[1, num_steps:, :] = 0
test_data[2, num_steps:, :] = 0
test_data[4, num_steps:, :] = 0
i = 0
print("Generating...")
while i + num_steps < len(test_data[0]):
    x = np.zeros((1, num_steps, NUMBER_FEATURES), dtype=np.bool)
    x[0, :, :] = test_data[3, i : i + num_steps, :vocabulary]
    #for inst_num in range(test_data.shape[0]):5
    #    x[:, :, inst_num*NUMBER_FEATURES : ((inst_num+1)*NUMBER_FEATURES)] = test_data[inst_num, i : i + num_steps, :]
    if not x[0][0].max(): 
        i += 1
        continue
    test = x[0][0]
    prediction = model.predict(x)
    #predict_note = sample(prediction[0, num_steps - 1, :], diversity)'
    predict_note = np.zeros(NUMBER_FEATURES * INSTRUMENTS)
    predict_note[prediction[0, num_steps - 1, :] >= 0.01] = 1
    #test = sample(prediction)
    print([i for i in range(x[0, num_steps - 1, :].shape[0]) if x[0, num_steps - 1, :][i]])
    print([i for i in range(len(predict_note)) if predict_note[i]])
    for j in range(len(predict_note)):
        if predict_note[j]:
            test_data[math.floor(predict_note[j] / NUMBER_FEATURES), i + num_steps - 1, int(predict_note[j]) % NUMBER_FEATURES] = 1

    i += 1     
print("Done generating.")
data_new = np.copy(test_data)
data_new[1, :, :] = test_data[1, :, :]

generate_midi(data_new, 'test.mid')
print("Wrote to midi file.")

