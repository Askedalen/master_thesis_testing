import pyo
from random import random
import numpy as np
import config as conf
from MusicGenerator import MusicGenerator

class Synth:
    def __init__(self, s, transpo=1, mul=1, num_channels=10):
        self.server = s
        self.transpo = pyo.Sig(transpo)
        self.num_channels = num_channels
        self.prev_step = {'piano':[],'guitar':[],'bass':[],'drums':[]}

        self.notes = pyo.Notein(poly=self.num_channels, scale=0, first=0, last=127, channel=1)
        self.pit = self.notes["pitch"]
        self.pitHz = pyo.MToF(self.pit) * self.transpo
        self.amp = pyo.MidiAdsr(self.notes["velocity"], attack=0.001, decay=0.1, sustain=0.7, release=1, mul=0.1,)

        self.piano_notes = pyo.Notein(scale=0, first=0, last=127, channel=2)
        self.piano_amp = pyo.MidiAdsr(self.piano_notes["velocity"])
        self.piano_pitch = pyo.MToF(self.piano_notes["pitch"])
        self.piano_osc = pyo.Osc(pyo.SquareTable(), freq=self.piano_pitch, mul=self.piano_amp).mix(1)
        self.piano_rev = pyo.STRev(self.piano_osc, revtime=1, cutoff=4000, bal=0.2).out()
        self.piano_notes.keyboard('test')

        self.guitar_notes = pyo.Notein(scale=0, first=0, last=127, channel=3)
        self.guitar_amp = pyo.MidiAdsr(self.guitar_notes["velocity"])
        self.guitar_pitch = pyo.MToF(self.guitar_notes["pitch"])
        self.guitar_osc = pyo.Osc(pyo.SquareTable(), freq=self.guitar_pitch, mul=self.guitar_amp).mix(1)
        self.guitar_rev = pyo.STRev(self.guitar_osc, revtime=1, cutoff=4000, bal=0.2).out()

        self.bass_notes = pyo.Notein(scale=0, first=0, last=127, channel=4)
        self.bass_amp = pyo.MidiAdsr(self.bass_notes["velocity"])
        self.bass_pitch = pyo.MToF(self.bass_notes["pitch"])
        self.bass_osc = pyo.Osc(pyo.SquareTable(), freq=self.bass_pitch, mul=self.bass_amp).mix(1)
        self.bass_rev = pyo.STRev(self.bass_osc, revtime=1, cutoff=4000, bal=0.2).out()

        self.drums_notes = pyo.Notein(scale=0, first=0, last=127, channel=5)
        self.drums_amp = pyo.MidiAdsr(self.drums_notes["velocity"])
        self.drums_pitch = pyo.MToF(self.drums_notes["pitch"])
        self.drums_osc = pyo.Osc(pyo.SquareTable(), freq=self.drums_pitch, mul=self.drums_amp).mix(1)
        self.drums_rev = pyo.STRev(self.drums_osc, revtime=1, cutoff=4000, bal=0.2).out()

        # Anti-aliased stereo square waves, mixed from 10 streams to 1 stream
        # to avoid channel alternation on new notes.
        self.osc1 = pyo.LFO(self.pitHz, sharp=0.5, type=2, mul=self.amp).mix(1)
        self.osc2 = pyo.LFO(self.pitHz * 0.997, sharp=0.5, type=2, mul=self.amp).mix(1)

        # Stereo mix.
        self.mix = pyo.Mix([self.osc1, self.osc2], voices=2)

        # High frequencies damping.
        self.damp = pyo.ButLP(self.mix, freq=5000)

        # Moving notches, using two out-of-phase sine wave oscillators.
        self.lfo = pyo.Sine(0.2, phase=[random(), random()]).range(250, 4000)
        self.notch = pyo.ButBR(self.damp, self.lfo, mul=mul)

        self.current_noteons = np.zeros(conf.num_notes)
        self.current_noteoffs = np.zeros(conf.num_notes)

        self.tempo = conf.tempo
        self.beat_length = 1#60/self.tempo/conf.subdivision
       
        self.pat = pyo.Pattern(self._timestep, self.beat_length)

        self.generator = MusicGenerator()
        #self.testtrig = pyo.TrigFunc(self.test['trigon'], self._test, arg=list(range(10)))

    def _test(self, m):
        print(m)
        

    def out(self):
        "Sends the synth's signal to the audio output and return the object itself."
        self.notch.out()
        return self

    def sig(self):
        "Returns the synth's signal for future processing."
        return self.notch

    def keyboard(self):
        self.notes.keyboard()

    def start_playing(self):
        self.tfon = pyo.TrigFunc(self.notes["trigon"], self._noteon, arg=list(range(10)))
        self.tfoff = pyo.TrigFunc(self.notes["trigoff"], self._noteoff, arg=list(range(10)))
        #self.pat.play()

    def stop_playing(self):
        self.pat.stop()
        del self.tfon
        del self.tfoff

    def _noteon(self, voice):
        pitch = int(self.pit.get(all=True)[voice])
        if pitch >= conf.pr_start_idx and pitch <= conf.pr_end_idx:
            self.current_noteons[pitch - conf.pr_start_idx] = 1

    def _noteoff(self, voice):
        pitch = int(self.pit.get(all=True)[voice])
        if pitch >= conf.pr_start_idx and pitch <= conf.pr_end_idx:
            self.current_noteoffs[pitch - conf.pr_start_idx] = 1

    def _timestep(self):
        # This method is run for each 16th note
        # Fetches the current played notes, 
        # removes the notes that have been released since the last time step
        # and calls the model to generate accompaniment for the next timestep

            
        timestep = self.current_noteons
        noteons = np.where(self.current_noteons >= 1)
        if len(noteons[0]) > 0:
            for i in np.nditer(noteons):
                if self.current_noteoffs[i] >= 1:
                    self.current_noteons[i] = 0
                    self.current_noteoffs[i] = 0
        # Call ML-model with current and previous timesteps and recieve MIDI to play
        next_step = self.generator.step(timestep)
        #next_step = np.zeros((204))
        """ if 48 not in self.prev_step['piano']:
            next_step[24] = 1 """
        """ if len(self.prev_step['piano']) == 0:
            next_step[0] = 1
            next_step[24] = 1
            next_step[28]= 1
            next_step[84] = 1
            next_step[91] = 1
            next_step[144] = 1 """

        next_piano  = np.asarray(np.where(next_step[                 : conf.num_notes  ] >= 1))[0] + conf.pr_start_idx
        next_guitar = np.asarray(np.where(next_step[conf.num_notes   : conf.num_notes*2] >= 1))[0] + conf.pr_start_idx
        next_bass   = np.asarray(np.where(next_step[conf.num_notes*2 : conf.num_notes*3] >= 1))[0] + conf.pr_start_idx
        next_drums  = np.asarray(np.where(next_step[conf.num_notes*3 :                 ] >= 1))[0] + conf.pr_start_idx

        messages = [[],[],[]]

        for note in next_piano:
            if note not in self.prev_step['piano']:
                messages[0].append(144)
                messages[1].append(note)
                messages[2].append(100)
            print('noteon',messages)
        for note in self.prev_step['piano']:
            if note not in next_piano:
                messages[0].append(128)
                messages[1].append(note)
                messages[2].append(0)
            print('noteoff',messages)

        self.server.addMidiEvent(*messages)
        # TODO: Try using OSCListRecieve and OscSend

        self.prev_step = {
            'piano':next_piano,
            'guitar':next_guitar,
            'bass':next_bass,
            'drums':next_drums
        }

if __name__ == "__main__":
    s = pyo.Server()
    s.setMidiInputDevice(99)  # Open all input devices.
    s.boot()

    # Create the midi synth.
    a1 = Synth(s).out()
    a1._timestep()
    #a1._timestep()
    
    a1.keyboard()
    a1.start_playing()

    s.start()
    s.gui(locals())