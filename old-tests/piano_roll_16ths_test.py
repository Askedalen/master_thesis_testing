from __future__ import division
import pretty_midi as pm
import sys
import argparse
import numpy as np
import librosa
import os

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0, is_drum=False):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(program=program, is_drum=is_drum)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pm.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    midi.instruments.append(instrument)
    return instrument

# Convert MIDI to Piano Roll with sample rate equal to 32nd notes
midi = pm.PrettyMIDI("lmd_matched\C\B\Z\TRCBZKU12903CB7DEA\9d2fbf358d875f5af0dc011a8370236a.mid")

initial_tempo = midi.get_tempo_changes()[1][0] 
beat_length_seconds = 60 / initial_tempo # Get length of each beat in seconds
sampling_freq = 1 / (beat_length_seconds /8) # Calculate sampling frequency to get 32nd-notes

piano_rolls = []
for inst in midi.instruments:
    is_drum = inst.is_drum
    #if is_drum: #<-- include drum tracks in the piano roll
    #    inst.is_drum = False
    piano_rolls.append([inst.get_piano_roll(fs=sampling_freq), inst.program, is_drum])


#Return to MIDI files
midis = []
for pr in piano_rolls:
    pr[0][pr[0] > 127] = 127
    midis.append(piano_roll_to_pretty_midi(pr[0], sampling_freq, pr[1], pr[2]))

new_midi = pm.PrettyMIDI(resolution=midi.resolution, initial_tempo=initial_tempo)
new_midi.instruments += midis

if os.path.exists('original.mid'):
    os.remove('original.mid')
if os.path.exists('parsed.mid'):
    os.remove('parsed.mid')
midi.write('original.mid')
new_midi.write('parsed.mid')
print()