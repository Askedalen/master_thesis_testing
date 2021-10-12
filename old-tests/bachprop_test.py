import pygame
import numpy as np
import pretty_midi as pm


def parseMIDI(midi_file_path):
    """
    Process the MIDI in midi_file_path to extract
    the sequence of notes (dT, T, P). Timing is in 
    MIDI tick representation. The function also 
    returns the tick per beat (tpb) metaparameter
    """ 
    s = pm.PrettyMIDI(midi_file_path)
    tpb = float(s.resolution)
    events = mergeTrack(s)
    T = []
    P = []
    dT = []
    dt = 0
    for n, event in enumerate(events):
        if event.name == 'Note On' and event.data[1] > 0:
            pitch_n = event.data[0]
            n2 = n
            duration_n = 0
            while True:
                n2 += 1
                if n2 > (len(events)-1):
                    break
                duration_n += events[n2].tick
                if events[n2].data[0] == pitch_n and events[n2].name == 'Note Off':
                    break
                if events[n2].data[0] == pitch_n and events[n2].name == 'Note On' and events[n2].data[1] == 0:
                    break
            if duration_n > 0.:
                P.append(pitch_n)
                T.append(duration_n)
                dT.append(event.tick+dt)
            dt = 0
        elif event.name == 'Note Off' or event.data[1] == 0:
            dt += event.tick
    
    #Tick (integer) to beat fraction (float)
    dT = [float(dt)/tpb for dt in dT]
    T = [float(t)/tpb for t in T]
    return dT, T, P, tpb

test = parseMIDI("lmd_matched\A\A\A\TRAAAGR128F425B14B\\1d9d16a9da90c090809c153754823c2b.mid")
print(test)