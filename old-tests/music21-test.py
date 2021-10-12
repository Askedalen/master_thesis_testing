import os
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    
    path = "lmd_matched"
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.mid' in file:
                files.append(os.path.join(r, file))

    midi_data = []
    i = 0
    startTime = time.time()
    for file in files:
        if i >= 100: break
        try:
            midi_data.append(pm.PrettyMIDI(file))
            i += 1
        except:
            print("Could not load file {}".format(file))

    endTime = time.time()
    print("Loaded data in {} seconds.".format(endTime - startTime))

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes