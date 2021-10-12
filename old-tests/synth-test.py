import pygame
import pygame.midi
from time import sleep
import sys
 
    
C = 74
MAX = 127
brief = .5
 
def midi(note=[C], volume=MAX, length=brief):
    for n in note:
        midi_out.note_on(n, volume) # 74 is middle C, 127 is "how loud" - max is 127
    sleep(brief)
    for n in note: 
        midi_out.note_off(n, volume)
    sleep(brief)
 
 
#                init
# =======================================
GRAND_PIANO = 0
CHURCH_ORGAN = 19
instrument = CHURCH_ORGAN
pygame.init()
pygame.midi.init()
port = pygame.midi.get_default_output_id()
midi_out = pygame.midi.Output(port, 0)
midi_out.set_instrument(instrument)
print ("using output_id :%s:" % port)
# =======================================
 
def exit():
    global midi_out, music
 
    music = 0
    del midi_out
    pygame.midi.quit()
    pygame.quit()
    sys.exit()
 
CM = [74, 78, 81]
D = [74, 76, 81]
FM = [72, 76, 79]
screen = pygame.display.set_mode((400, 400))
music = 1
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                exit()
 
    if music:
        seq = CM, D, CM, FM
        for s in seq:
            midi(s)