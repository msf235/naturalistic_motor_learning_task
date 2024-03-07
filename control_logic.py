import pygame
import sys
import numpy as np


class inputHandler:
    def __init__(self):
        pass

    def event_handler(self, events):
        for event in events:
            self.process_event(event)

    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            self.process_keydown(event.key)


class simulationStateHandler(inputHandler):
    def __init__(self):
        super().__init__()
        self.paused = False

    def process_keydown(self, key):
        if key == pygame.K_q:
            print('Quitting.')
            sys.exit()
        elif key == pygame.K_SPACE:
            print("Toggling pause.")
            self.paused = not self.paused


class modelControlHandler(inputHandler):
    def __init__(self, model, data, gain=.1):
        super().__init__()
        # self.model = model
        # self.data = data
        self.gain = gain
        self.ctrl = np.zeros(data.ctrl.shape)
        self.keys = [pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f,
                     pygame.K_g, pygame.K_h, pygame.K_j, pygame.K_k]
        self.pos_keys = [pygame.K_a, pygame.K_d, pygame.K_g, pygame.K_j]
        self.neg_keys = [pygame.K_s, pygame.K_f, pygame.K_h, pygame.K_k]
        self.key_nums = {pygame.K_a: 0, pygame.K_s: 0, pygame.K_d: 1,
                             pygame.K_f: 1, pygame.K_g: 2, pygame.K_h: 2,
                             pygame.K_j: 3, pygame.K_k: 3}

    def process_keydown(self, key):
        if key in self.pos_keys:
            self.ctrl[self.key_nums[key]] += self.gain
        if key in self.neg_keys:
            self.ctrl[self.key_nums[key]] -= self.gain
