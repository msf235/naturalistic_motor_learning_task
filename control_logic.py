import pygame
import sys

class pauseState:
    def __init__(self):
        self.paused = False
    def keypause(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused

class inputHander:
    def __init__(self, model, data, gain=.1):
        self.model = model
        self.data = data
        self.paused = False
        self.gain = gain
        # self.key_tuples = [(a,pygame.K_a), (s,pygame.K_s), (d,pygame.K_d),
                             # (f,pygame.K_f), (g,pygame.K_g), (h,pygame.K_h),
                             # (j,pygame.K_j), (k,pygame.K_k), (l,pygame.K_l)]
        self.keys = [pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f,
                     pygame.K_g, pygame.K_h, pygame.K_j, pygame.K_k]
        self.pos_keys = [pygame.K_a, pygame.K_d, pygame.K_g, pygame.K_j]
        self.neg_keys = [pygame.K_s, pygame.K_f, pygame.K_h, pygame.K_k]
        self.key_nums = {pygame.K_a: 0, pygame.K_s: 0, pygame.K_d: 1,
                             pygame.K_f: 1, pygame.K_g: 2, pygame.K_h: 2,
                             pygame.K_j: 3, pygame.K_k: 3}
        # self.key_tuples = self.key_tuples[:model.nq]
        # self.key_conv = dict(self.key_tuple)
        # self.key_conv = dict(a=pygame.K_a, s=pygame.K_s, d=pygame.K_d,
                             # f=pygame.K_f, g=pygame.K_g, h=pygame.K_h,
                             # j=pygame.K_j, k=pygame.K_k)
        # self.key_nums = [(key, k) for k, key in enumerate(self.key_tuples)]
        # self.key_nums = dict((key[1], k) for k, key in enumerate(self.key_tuples))

        # self.pos_keys = ['a', 'd', 'g', 'j']
        # self.pos_keys = [self.key_conv[key] for key in self.pos_keys]
        # self.neg_keys = ['s', 'f', 'h', 'k']
        # self.neg_keys = [self.key_conv[key] for key in self.neg_keys]

    def key_convert(self, key):
        if key in self.key_conv:
            return self.key_conv[key]
        return None

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
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in self.pos_keys:
            self.data.ctrl[self.key_nums[key]] += self.gain
        elif key in self.neg_keys:
            self.data.ctrl[self.key_nums[key]] -= self.gain

