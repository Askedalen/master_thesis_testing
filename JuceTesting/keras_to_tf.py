from keras.models import load_model
import tensorflow as tf

#TODO: Store the models and weights seperately so that prediciton can be done with a different batch size
chord_model = load_model('JuceTesting/chord_best.hdf5')
poly_model = load_model('JuceTesting/poly_best.hdf5')
chord_model.save('JuceTesting/chord.pb', save_format='tf')
poly_model.save('JuceTesting/poly.pb', save_format='tf')

