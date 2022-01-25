import pyo
from random import random
import numpy as np
import config as conf
from MusicGenerator import MusicGenerator

class Synth:
    def __init__(self, transpo=1, mul=1, num_channels=10):
        self.transpo = pyo.Sig(transpo)
        self.num_channels = num_channels
        
        self.notes = pyo.Notein(poly=self.num_channels, scale=0, first=0, last=127)
        self.pit = self.notes["pitch"]
        self.pitHz = pyo.MToF(self.pit) * self.transpo
        self.amp = pyo.MidiAdsr(self.notes["velocity"], attack=0.001, decay=0.1, sustain=0.7, release=1, mul=0.1,)

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
        self.beat_length = 60/self.tempo/conf.subdivision
       
        self.pat = pyo.Pattern(self._timestep, self.beat_length)

        self.generator = MusicGenerator()

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
        self.pat.play()

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
        # Essentially quantizes what is played by the user

        noteons = np.where(self.current_noteons >= 1)
        if len(noteons[0]) > 0:
            for i in np.nditer(noteons):
                if self.current_noteoffs[i] >= 1:
                    self.current_noteons[i] = 0
                    self.current_noteoffs[i] = 0
            
        timestep = self.current_noteons
        # TODO: Call ML-model with current and previous timesteps and recieve MIDI to play
        next_step = self.generator.step(timestep)

if __name__ == "__main__":
    s = pyo.Server()
    s.setMidiInputDevice(99)  # Open all input devices.
    s.boot()

    # Create the midi synth.
    a1 = Synth().out()
    
    a1.keyboard()
    a1.start_playing()

    s.start()
    s.gui(locals())