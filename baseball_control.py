import mujoco as mj
from mujoco import viewer
import time
import sys
import pygame
import control_logic as cl

pygame.init()
window = pygame.display.set_mode((300, 300))
clock = pygame.time.Clock()

# xml_file = 'arm.xml'
xml_file = 'humanoid_and_baseball.xml'
with open(xml_file, 'r') as f:
  xml = f.read()


model = mj.MjModel.from_xml_string(xml)
data = mj.MjData(model)

inp_handler = cl.inputHander(model, data)

with mj.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if not inp_handler.paused:
            mj.mj_step(model, data)
            viewer.sync()
        inp_handler.event_handler(pygame.event.get())
        time.sleep(.007)
