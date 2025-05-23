import pyo
from random import random
import numpy as np
import config as conf
from MusicGenerator import MusicGenerator
import time

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

        self.piano_osc = pyo.RCOsc(freq=[i for i in range(10)], mul=0.1)
        self.piano_mix = self.piano_osc.mix(1)
        self.piano_rev = pyo.STRev(self.piano_mix, revtime=1, cutoff=4000, bal=0.2).out()

        self.guitar_osc = pyo.RCOsc([i for i in range(10)], mul=0.1)
        self.guitar_mix = self.guitar_osc.mix(1)
        self.guitar_rev = pyo.STRev(self.guitar_mix, revtime=1, cutoff=4000, bal=0.2).out()

        self.bass_osc = pyo.RCOsc([i for i in range(10)], mul=0.1)
        self.bass_mix = self.bass_osc.mix(1)
        self.bass_rev = pyo.STRev(self.bass_mix, revtime=1, cutoff=4000, bal=0.2).out()

        self.drums_osc = pyo.RCOsc([i for i in range(10)], mul=0.1)
        self.drums_mix = self.drums_osc.mix(1)
        self.drums_rev = pyo.STRev(self.drums_mix, revtime=1, cutoff=4000, bal=0.2).out()

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
        self.beat_length = 60/self.tempo#/conf.subdivision
       
        self.gen_pat = pyo.Pattern(self._timestep, self.beat_length)
        self.play_pat = pyo.Pattern(self._timestep_play, self.beat_length)

        self.generator = MusicGenerator()
        self.num_steps = 0

        self.piano_freqs = []
        self.guitar_freqs = []
        self.bass_freqs = []
        self.drums_freqs = []
        
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
        self.gen_pat.play()
        time.sleep(0.01)
        self.play_pat.play()

    def stop_playing(self):
        self.gen_pat.stop()
        self.play_pat.stop()
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

    def _timestep_play(self):
        if len(self.piano_freqs) > 0:
            self.piano_osc.setFreq(self.piano_freqs)
            if not self.piano_osc.isPlaying():
                self.piano_osc.play()
            print("Playing: ", self.piano_freqs)
        else:
            if self.piano_osc.isPlaying():
                self.piano_osc.stop()

        if len(self.guitar_freqs) > 0:
            self.guitar_osc.setFreq(self.guitar_freqs)
            if not self.guitar_osc.isPlaying():
                self.guitar_osc.play()
        else:
            if self.guitar_osc.isPlaying():
                self.guitar_osc.stop()

        if len(self.bass_freqs) > 0:
            self.bass_osc.setFreq(self.bass_freqs)
            if not self.bass_osc.isPlaying():
                self.bass_osc.play()
        else:
            if self.bass_osc.isPlaying():
                self.bass_osc.stop()
            
        if len(self.drums_freqs) > 0:
            self.drums_osc.setFreq(self.drums_freqs)
            if not self.drums_osc.isPlaying():
                self.drums_osc.play()
        else:
            if self.drums_osc.isPlaying():
                self.drums_osc.stop()

    def _timestep(self):
        # This method is run for each 16th note
        # Fetches the current played notes, 
        # removes the notes that have been released since the last time step
        # and calls the model to generate accompaniment for the next timestep
        self.num_steps += 1
        #print('Time step', self.num_steps)
        timestep = self.current_noteons
        noteons = np.where(self.current_noteons >= 1)
        if len(noteons[0]) > 0:
            for i in np.nditer(noteons):
                if self.current_noteoffs[i] >= 1:
                    self.current_noteons[i] = 0
                    self.current_noteoffs[i] = 0
        # Call ML-model with current and previous timesteps and recieve MIDI to play
        next_step = self.generator.step(timestep)

        next_piano  = np.asarray(np.where(next_step[                 : conf.num_notes  ] >= 1))[0] + conf.pr_start_idx
        next_guitar = np.asarray(np.where(next_step[conf.num_notes   : conf.num_notes*2] >= 1))[0] + conf.pr_start_idx
        next_bass   = np.asarray(np.where(next_step[conf.num_notes*2 : conf.num_notes*3] >= 1))[0] + conf.pr_start_idx
        next_drums  = np.asarray(np.where(next_step[conf.num_notes*3 :                 ] >= 1))[0] + conf.pr_start_idx

        self.piano_freqs = []
        self.guitar_freqs = []
        self.bass_freqs = []
        self.drums_freqs = []

        for midi_note in next_piano:
            self.piano_freqs.append(pyo.midiToHz(midi_note))
        for midi_note in next_guitar:
            self.guitar_freqs.append(pyo.midiToHz(midi_note))
        for midi_note in next_bass:
            self.bass_freqs.append(pyo.midiToHz(midi_note))
        for midi_note in next_drums:
            self.drums_freqs.append(pyo.midiToHz(midi_note))
        
        

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
    
    a1.keyboard()
    a1.start_playing()

    s.start()
    s.gui(locals())