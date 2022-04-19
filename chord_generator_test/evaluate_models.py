from keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
import numpy as np
import load_data as load
import config as conf
import math
import _pickle as pickle
import midi_functions as mf
import pretty_midi
import os
import time
from MusicGenerator import MusicGenerator, BaselineMusicGenerator
np.random.seed(2021)

if __name__ == '__main__':
    print('Evaluating chord/poly model')
    start_time = time.process_time()
    chord_model_filename = ""
    poly_model_filename = ""
    baseline_model_filename = ""
    chord_model = load_model(chord_model_filename)
    poly_model = load_model(poly_model_filename)

    print('Loading test data...')
    melodies, targets = load.load_test_data(num_songs=10)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    generator = MusicGenerator(chord_model=chord_model, poly_model=poly_model, binary=True)
    for i in range(len(melodies)):
        generator.reset()
        melody = melodies[i]
        target = targets[i]

        output = []
        for j in range(len(melody)):
            timestep = melody[j]
            out = generator.step(timestep)
            for x in range(len(out)):
                if out[x] and target[x]:
                    tp += 1
                elif not out[x] and not target[x]:
                    tn += 1
                elif out[x] and not target[x]:
                    fp += 1
                elif not out[x] and target[x]:
                    fn += 1
    
    total_preds = tp + tn + fp + fn
    acc = (tp+tn)/total_preds
    prec = tp/(tp + fp)
    rec = tp/(tp + fn)

    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('Total time:', time.process_time() - start_time)
    print()

    print('Evaluating baseline model')
    b_start_time = time.process_time()
    b_tp = 0
    b_tn = 0
    b_fp = 0
    b_fn = 0
    baseline_model = load_model(baseline_model_filename)

    baseline_generator = BaselineMusicGenerator(baseline_model, binary=True)
    for i in range(len(melodies)):
        baseline_generator.reset()
        melody = melodies[i]
        target = targets[i]

        output = []
        for j in range(len(melody)):
            timestep = melody[j]
            out = baseline_generator.step(timestep)
            for x in range(len(out)):
                if out[x] and target[x]:
                    b_tp += 1
                elif not out[x] and not target[x]:
                    b_tn += 1
                elif out[x] and not target[x]:
                    b_fp += 1
                elif not out[x] and target[x]:
                    b_fn += 1
    
    total_preds = b_tp + b_tn + b_fp + b_fn
    b_acc = (b_tp+b_tn)/total_preds
    b_prec = b_tp/(b_tp + b_fp)
    b_rec = b_tp/(b_tp + b_fn)

    print('Accuracy:', b_acc)
    print('Precision:', b_prec)
    print('Recall:', b_rec)
    print('Total time:', time.process_time() - b_start_time)
    print()
