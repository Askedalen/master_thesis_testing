from keras.models import load_model
import parse_MIDI_test as parse

model = load_model('lstm_test_0_model.md5')

data = parse.getAll(1)
data_t = data[1]

data_t = data_t.reshape((1, data_t.shape[0], data_t.shape[1]))
input_data = data_t[:, :, 64]
prediciton = model.predict(input_data)

print()
