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

class inputHander:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.paused = False
        self.nq = model.nq  # Number of actuators

    def event_handler(self, events):
        for event in events:
            self.process_event(event)

    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            self.process_keydown(event.key)

    def process_keydown(self, key):
        if key == pygame.K_q:
            print('Quitting')
            sys.exit()
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        if key == pygame.K_a:
            self.data.ctrl[0] -= 50
        if key == pygame.K_s:
            self.data.ctrl[0] += 50
        if key == pygame.K_d:
            self.data.ctrl[1] -= 50
        if key == pygame.K_f:
            self.data.ctrl[1] += 50


model = mj.MjModel.from_xml_string(xml)
data = mj.MjData(model)

inp_handler = inputHander(model, data)

with mj.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if not inp_handler.paused:
            mj.mj_step(model, data)
            viewer.sync()
        inp_handler.event_handler(pygame.event.get())
        time.sleep(.007)

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
