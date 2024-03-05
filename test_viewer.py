import mujoco as mj
from mujoco import viewer
import time
import sys
import pygame
# from pynput.keyboard import Listener
# import termios, fcntl, sys, os
# from pynput import keyboard

pygame.init()
window = pygame.display.set_mode((300, 300))
clock = pygame.time.Clock()

breakpoint()

xml_file = 'arm.xml'
with open(xml_file, 'r') as f:
  xml = f.read()
del f

class keyRecord:
    def __init__(self):
        self.key = None
    def record(self, keycode):
        self.key = chr(keycode)

class pauseState:
    def __init__(self):
        self.paused = False
    def keypause(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused

model = mj.MjModel.from_xml_string(xml)
data = mj.MjData(model)

recorder = keyRecord()
pauser = pauseState()

pause = False
with mj.viewer.launch_passive(model, data, key_callback=pauser.keypause) \
        as viewer:
    while viewer.is_running():
        if not pauser.paused:
            mj.mj_step(model, data)
            viewer.sync()
        time.sleep(.007)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            print(pygame.key.name(event.key))

sys.exit()


























# class controlState:
    # def __init__(self, model, data):
        # self.paused = False
        # self.model = model
        # self.data = data
        # self.crankenable = False
    # def keypause(self, keycode):
        # if chr(keycode) == ' ':
            # self.paused = not self.paused
    # def crank(self, keycode):
        # if chr(keycode) == ' ':
            # self.crankenable = True
        # else:
        # if self.crankenable:
            # self.data.ctrl[0] = -500
        # self.crankenable = False



model = mj.MjModel.from_xml_string(xml)
data = mj.MjData(model)

# pauser = controlState(model, data)

def test_control(model, data):
    # data.ctrl[0] -= .01
    data.ctrl[0] = -400

# mj.set_mjcb_control(test_control)

# viewer.launch_passive(model, data, key_callback=key_callback)

def on_press(key):
    print("Key pressed")

def on_release(key):
    print("Key released")

# with Listener(on_press=on_press, on_release=on_release) as listener:
    # listener.join()

clock = pygame.time.Clock()
pause = False
pygame.init()
# sys.exit()
run = True

with mj.viewer.launch_passive(model, data,
                                  # key_callback=pauser.keypause
                                  # key_callback=pauser.crank
                                 ) as viewer:
    while viewer.is_running():
        try:
            c = sys.stdin.read(1)
            if c:
                if c == ' ':
                    pause = not pause
                print("Got character", repr(c))
        except IOError: pass

        # sys.exit()
        # for event in pygame.event.get():
            # if event.type == pygame.KEYDOWN:
                # if event.key == pygame.K_SPACE:
                    # pause = not pause
        # event = events.get(.003)
        time.sleep(.003)
        # if event is None:
            # print('You did not press a key within one second')
        # else:
            # print('Received event {}'.format(event))

        # pause = pygame.key.get_pressed()[pygame.K_SPACE]
        # if not pauser.paused:
        if not pause:
            mj.mj_step(model, data)
            viewer.sync()

del model, data
