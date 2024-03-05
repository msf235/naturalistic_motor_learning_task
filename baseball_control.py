import mujoco as mj
from mujoco import viewer
import time
import sys
import pygame
import numpy as np
import scipy
import control_logic as cl

pygame.init()
# window = pygame.display.set_mode((300, 300))
clock = pygame.time.Clock()

# xml_file = 'arm.xml'
xml_file = 'humanoid_and_baseball.xml'
with open(xml_file, 'r') as f:
  xml = f.read()


model = mj.MjModel.from_xml_string(xml)
data = mj.MjData(model)
mj.mj_forward(model, data)

# Burn in:
for k in range(10):
    mj.mj_step(model, data)

nq = model.nq
nu = model.nu

R = np.eye(nu)

jac_com = np.zeros((3, nq))
mj.mj_jacSubtreeCom(model, data, jac_com, model.body('torso').id)

# Get the Jacobian for the left foot.
jac_lfoot = np.zeros((3, nq))
mj.mj_jacBodyCom(model, data, jac_lfoot, None, model.body('foot_left').id)
jac_rfoot = np.zeros((3, nq))
mj.mj_jacBodyCom(model, data, jac_rfoot, None, model.body('foot_right').id)
jac_base = (jac_lfoot + jac_rfoot) / 2

jac_diff = jac_com - jac_base
Qbalance = jac_diff.T @ jac_diff

Q = 1000*np.block([[Qbalance, np.zeros((nq, nq))],
              [np.zeros((nq, nq)), np.eye(nq)]])

A = np.zeros((2*nq, 2*nq))
B = np.zeros((2*nq, nu))
epsilon = 1e-6
flg_centered = True
mj.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

breakpoint()

jac_diff = jac_com - jac_foot
Qbalance = jac_diff.T @ jac_diff

inp_handler = cl.inputHander(model, data)
inp_handler.paused = True


with mj.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 10
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 180
    while viewer.is_running():
        if not inp_handler.paused:
            mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
            dx = np.hstack((dq, data.qvel)).T
            mj.mj_step(model, data)
            viewer.sync()
        inp_handler.event_handler(pygame.event.get())
        time.sleep(.007)
