import os

testing         = True
random_data     = False

data_dir        = os.path.join('chord_generator_test', 'data')
results_dir     = os.path.join('chord_generator_test', 'results')
chord_model_dir = os.path.join(results_dir, 'models', 'chord')
poly_model_dir  = os.path.join(results_dir, 'models', 'poly')
random_data_dir = os.path.join(data_dir, 'random')
midi_unmod_dir  = os.path.join(data_dir, 'midi_unmod')
midi_mod_dir    = os.path.join(data_dir, 'midi_mod')
chord_dir       = os.path.join(data_dir, 'chords')
melody_dir      = os.path.join(data_dir, 'melodies')
instrument_dir  = os.path.join(data_dir, 'instruments')

subdivision     = 4 # Subdivision for each beat
num_notes       = 60
pr_start_idx    = 24
pr_end_idx      = 84
