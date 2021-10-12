import sys, pygame.midi
import time

pygame.midi.init()

num_devices = pygame.midi.get_count()

for m in range(num_devices):
    print("Device {}".format(m))
    print(str(pygame.midi.get_device_info(m)))

print("Ouput is {}".format(pygame.midi.get_default_output_id()))



midiInput = pygame.midi.Input(1)
midiOutput = pygame.midi.Output(0,0)
midiOutput.set_instrument(0)
while 1:
    while(pygame.midi.Input.poll(midiInput) == False):
        time.sleep(0.0001)
    midi_data = pygame.midi.Input.read(midiInput,1)
    midi_note, timestamp = midi_data[0]
    note_status, keynum, velocity, unused = midi_note
    midiOutput.note_on(keynum, velocity)
    print("Midi Note: \n\tNote Status: ", note_status, " Key Number: ", keynum," Velocity: " , velocity, "\n\tTime Stamp: ", timestamp)
    if note_status == 144:
        key_down = True
    elif note_status == 128: 
        key_down = False
    else:
        print("Unknown status!")