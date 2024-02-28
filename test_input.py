# import pygame

# pygame.init()
# window = pygame.display.set_mode((300, 300))
# clock = pygame.time.Clock()

# rect = pygame.Rect(0, 0, 20, 20)
# rect.center = window.get_rect().center
# vel = 5

import termios, fcntl, sys, os
# fd = sys.stdin.fileno()

# oldterm = termios.tcgetattr(fd)
# newattr = termios.tcgetattr(fd)
# newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
# termios.tcsetattr(fd, termios.TCSANOW, newattr)

# oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
# fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

class switchKey:
    def __init__(self):
        self.keypressed = False
    def switch(self, key):
        c = sys.stdin.read(1)
        if c == key and self.keypressed == False:
            return True
        else:
            self.keypressed = False
            return False

switcher = switchKey()
running = True
# TODO: test key input on windows
while running:
    pause = switcher.switch(' ')
    if pause:
        print("Paused")
    else:
        print("Not paused")
    # try:
        # c = sys.stdin.read(1)
        # if c:
            # if c == 'j':
                # running = False
            # print("Got character", repr(c))
    # except IOError: pass

sys.exit()

try:
    while 1:
        try:
            c = sys.stdin.read(1)
            if c:
                print("Got character", repr(c))
        except IOError: pass
finally:
    termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
sys.exit()

run = True
while run:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            print(pygame.key.name(event.key))

    # keys = pygame.key.get_pressed()
    
    # rect.x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * vel
    # rect.y += (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * vel
        
    # rect.centerx = rect.centerx % window.get_width()
    # rect.centery = rect.centery % window.get_height()

    # window.fill(0)
    # pygame.draw.rect(window, (255, 0, 0), rect)
    # pygame.display.flip()

pygame.quit()
exit()
