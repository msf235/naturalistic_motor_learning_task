import mujoco as mj
from mujoco import viewer
import time
import sys
import pygame
import numpy as np
import scipy
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
mj.mj_forward(model, data)

nq = model.nq
nu = model.nu

# Burn in:
for k in range(10):
    mj.mj_step(model, data)

# Get initial stabilizing controls
data.qacc = 0
qpos0 = data.qpos.copy()  # Save the position setpoint.
mj.mj_inverse(model, data)
qfrc0 = data.qfrc_inverse.copy()
ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.

# Reset and burn in again:
mj.mj_resetData(model, data)
for k in range(10):
    mj.mj_step(model, data)

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

fact = 3
Qupright = np.zeros((nq, nq))
Qupright[0,0] = fact
Qupright[1,1] = fact

Q = fact*np.block([[0*Qbalance + Qupright, np.zeros((nq, nq))],
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

inp_handler = cl.inputHander(model, data)
inp_handler.paused = True

dq = np.zeros(nq)
qpos0 = data.qpos.copy()

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 10
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 180
    while viewer.is_running():
        if not inp_handler.paused:
            mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
            dx = np.hstack((dq, data.qvel)).T

            data.ctrl = ctrl0 - K @ dx

            mj.mj_step(model, data)
            viewer.sync()
        inp_handler.event_handler(pygame.event.get())
        time.sleep(.007)
