import os

testing         = True
random_data     = False

prefix_folder = 'chord_generator_test'
model_filenames = { 
    'chord_model':'models/chord_model.pb', 
    'chord_weights':'models/chord_weights.pb',
    'poly_model':'models/poly_model.pb',
    'poly_weights':'models/poly_weights.pb'
}

data_dir        = os.path.join(prefix_folder, 'data')
results_dir     = os.path.join(prefix_folder, 'results')
chord_model_dir = os.path.join(results_dir, 'models', 'chord')
poly_model_dir  = os.path.join(results_dir, 'models', 'poly')
music_gen_dir   = os.path.join(results_dir, 'generated_music')
random_data_dir = os.path.join(data_dir, 'random')
midi_unmod_dir  = os.path.join(data_dir, 'midi_unmod')
midi_mod_dir    = os.path.join(data_dir, 'midi_mod')
chord_dir       = os.path.join(data_dir, 'chords')
melody_dir      = os.path.join(data_dir, 'melodies')
instrument_dir  = os.path.join(data_dir, 'instruments')

num_steps       = 128
chord_interval  = 16
threshold       = 0.25

subdivision     = 4 # Subdivision for each beat
num_notes       = 60
pr_start_idx    = 24
pr_end_idx      = 84
tempo           = 120
