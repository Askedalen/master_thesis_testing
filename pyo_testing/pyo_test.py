from pyo import *
import time

s = Server().boot()
s.start()

notes = Notein()
pit = notes["pitch"]
pitHz = MToF(pit)
amp = MidiAdsr(notes["velocity"])
osc = Sine(freq=pitHz, mul=amp*0.5).mix(1).out()

notesArr = [60, 62, 64, 65, 67, 69, 71, 72]
currentNote = 0

def callback(a, args):
    global currentNote
    global s
    s.addMidiEvent([145, 129], [notesArr[currentNote%8], notesArr[(currentNote-1)%8]], [100, 0])
    currentNote += 1
    print('sending event', currentNote)  

    
test = OscDataReceive(9900, '/test', callback)
test2 = OscDataSend("i", 9900, '/test')

""" while currentNote < 16:
    callback(None, None)
    time.sleep(.5) """

def test1():
    test2.send([1])
pat = Pattern(test1, time=0.5).play()

s.gui(locals())
