from queue import Full
import config as conf
from test_configurations import FullModel
from keras.models import load_model
import os

if __name__ == '__main__':
    chord_config = {'batch_size':128,
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

    #model = FullModel(chord_config, poly_config)
    #model.create_empty_models(output_dir='pyo_testing')

    chord_model = load_model(os.path.join('pyo_testing', 'models', 'chord_model.pb'))
    poly_model = load_model(os.path.join('pyo_testing', 'models', 'poly_model.pb'))

    chord_model.load_weights(os.path.join('pyo_testing', 'models', 'chord_weights.pb'))
    poly_model.load_weights(os.path.join('pyo_testing', 'models', 'poly_weights.pb'))

    print('Done.')