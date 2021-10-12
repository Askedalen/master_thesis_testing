import pretty_midi
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt

inst_types = {'DRUM':0, 'BASS':1, 'GUITAR':2, 'PIANO':3, 'VOCAL':4, 'MELODY':5, 'OTHER':6}

def searchName(keys, strings):
    for s in strings:
        for k in keys:
            if k.lower() in s.lower():
                return True
    return False

def get_inst_type(inst_data):
    if inst_data.is_drum:
        return inst_types['DRUM']
    inst_name = pretty_midi.program_to_instrument_name(inst_data.program)
    track_name = inst_data.name

    if searchName(["mel", "melody"],        [inst_name, track_name]): return inst_types['MELODY']
    if searchName(["bass"],                 [inst_name, track_name]): return inst_types['BASS']
    if searchName(["guit"],                 [inst_name, track_name]): return inst_types['GUITAR']
    if searchName(["pian", "key", "organ"], [inst_name, track_name]): return inst_types['PIANO']
    if searchName(["voc", "voi", "choir"],  [inst_name, track_name]): return inst_types['VOCAL']

    return inst_types['OTHER']

if __name__ == "__main__":
    get_inst_type(None)