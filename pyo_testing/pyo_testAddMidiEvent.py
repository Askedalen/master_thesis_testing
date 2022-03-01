from pyo import *
import time

s = Server().boot()
s.start()

osc = RCOsc(freq=[440, 480, 500], mul=.1)
rev = STRev(osc)
f = rev.mix(1).out()

notesArr = [[60, 64], [60, 64, 67], [60, 67, 71], [60, 71, 74], []]
currentNote = 0

def callback(a, args):
    global currentNote
    freqs = []
    for note in notesArr[currentNote%len(notesArr)]:
        freqs.append(midiToHz(note))
    #freqs = [midiToHz(notesArr[currentNote%len(notesArr)][0]), midiToHz(notesArr[currentNote%len(notesArr)][1])]
    if len(freqs) > 0:
        osc.setFreq(freqs)
        if not osc.isPlaying():
            osc.play()
    else:
        osc.stop()
    currentNote += 1
    print('sending event', currentNote)  
    
test = OscDataReceive(9900, '/test', callback)
test2 = OscDataSend("i", 9900, '/test')

while currentNote < 16:
    callback(None, None)
    time.sleep(.5)

def test1():
    test2.send([1])
pat = Pattern(test1, time=0.5).play()

s.gui(locals())
