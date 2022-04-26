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
    if conf.testing:
        chord_model_filename = "chord_generator_test\\results\\tests\\full_model\models\chord_model.pb"
        poly_model_filename = "chord_generator_test\\results\\tests\\full_model\models\poly_model.pb"
        baseline_model_filename = "chord_generator_test\\results\\tests\\baseline_d220419_t1610/model.pb"
    else:
        chord_model_filename = "results/tests/d220421_t1123/models/chord_model.pb"
        poly_model_filename = "results/tests/d220422_t1154/models/poly_model.pb"
        baseline_model_filename = "results/tests/baseline_d220419_t1610/model.pb"
        
    chord_model = load_model(chord_model_filename)
    poly_model = load_model(poly_model_filename)

    threshold=0.08

    print('Loading test data...')
    melodies, targets = load.load_test_data(num_songs=0)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    print('Testing main model')
    generator = MusicGenerator(chord_model=chord_model, poly_model=poly_model, binary=True, threshold=threshold)
    batch_time = time.process_time()
    for i in range(len(melodies)):
        if i % 100 == 0:
            print(f'Song {i} of {len(melodies)}. {time.process_time() - batch_time} seconds since last update.') 
            batch_time = time.process_time()
        generator.reset()
        melody = melodies[i]
        target = targets[i]

        output = []
        for j in range(len(target)):
            timestep = melody[j]
            target_timestep = target[j]
            out = generator.step(timestep)
            for x in range(len(out)):
                if out[x] and target_timestep[x]:
                    tp += 1
                elif not out[x] and not target_timestep[x]:
                    tn += 1
                elif out[x] and not target_timestep[x]:
                    fp += 1
                elif not out[x] and target_timestep[x]:
                    fn += 1
    
    total_preds = tp + tn + fp + fn
    acc = (tp+tn)/total_preds
    prec = tp/(tp + fp)
    rec = tp/(tp + fn)

    print(tp, tn, fp, fn)
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

    baseline_generator = BaselineMusicGenerator(baseline_model, binary=True, threshold=threshold)
    batch_time = time.process_time()
    for i in range(len(melodies)):
        if i % 100 == 0:
            print(f'Song {i} of {len(melodies)}. {time.process_time() - batch_time} seconds since last update.') 
            batch_time = time.process_time()
        baseline_generator.reset()
        melody = melodies[i]
        target = targets[i]

        output = []
        for j in range(len(target)):
            timestep = melody[j]
            target_timestep = target[j]
            out = baseline_generator.step(timestep)
            for x in range(len(out)):
                if out[x] and target_timestep[x]:
                    b_tp += 1
                elif not out[x] and not target_timestep[x]:
                    b_tn += 1
                elif out[x] and not target_timestep[x]:
                    b_fp += 1
                elif not out[x] and target_timestep[x]:
                    b_fn += 1
    
    total_preds = b_tp + b_tn + b_fp + b_fn
    b_acc = (b_tp+b_tn)/total_preds
    b_prec = b_tp/(b_tp + b_fp)
    b_rec = b_tp/(b_tp + b_fn)

    print(b_tp, b_tn, b_fp, b_fn)
    print('Accuracy:', b_acc)
    print('Precision:', b_prec)
    print('Recall:', b_rec)
    print('Total time:', time.process_time() - b_start_time)
    print()
