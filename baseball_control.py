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

# Get all joint names.
joint_names = [model.joint(i).name for i in range(model.njnt)]

# Get indices into relevant sets of joints.
root_dofs = range(3)
body_dofs = range(3, nq)
abdomen_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if 'abdomen' in name
    and not 'z' in name
]
leg_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if ('hip' in name or 'knee' in name or 'ankle' in name)
    and not 'z' in name
]
balance_dofs = abdomen_dofs + leg_dofs
other_dofs = np.setdiff1d(body_dofs, balance_dofs)

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

# Cost coefficients.
BALANCE_COST        = 1000  # Balancing.
BALANCE_JOINT_COST  = 3     # Joints required for balancing.
OTHER_JOINT_COST    = .3    # Other joints.

# Construct the Qjoint matrix.
Qjoint = np.eye(nq)
Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

# Construct the Q matrix for position DoFs.
Qpos = BALANCE_COST * Qbalance + Qjoint

# No explicit penalty for velocities.
Q = np.block([[Qpos, np.zeros((nq, nq))],
              [np.zeros((nq, 2*nq))]])

# fact = 3
# Qupright = np.eye(nq, nq)
# # Qupright = np.zeros((nq, nq))
# Qupright[0,0] = fact
# Qupright[1,1] = fact

# Q = fact*np.block([[Qupright, np.zeros((nq, nq))],
              # [np.zeros((nq, nq)), np.eye(nq)]])

def get_feedback_ctrl_matrix(model, data, Q, R):
    nq = model.nq
    nu = model.nu
    A = np.zeros((2*nq, 2*nq))
    B = np.zeros((2*nq, nu))
    epsilon = 1e-6
    flg_centered = True
    mj.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

    # Solve discrete Riccati equation.
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K

dq = np.zeros(nq)

def get_lqr_ctrl(model, data, K):
    mj.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dx = np.hstack((dq, data.qvel)).T
    return -K @ dx

state_handler = cl.simulationStateHandler()
state_handler.paused = True

ctrl_handler = cl.modelControlHandler(model, data, gain=.1)

data.ctrl = ctrl0

K = get_feedback_ctrl_matrix(model, data, Q, R)

CTRL_STD = 0.05       # actuator units
CTRL_RATE = 0.8       # seconds
# Precompute some noise.
np.random.seed(1)
DURATION=500
nsteps = int(np.ceil(DURATION/model.opt.timestep))
perturb = np.random.randn(nsteps, nu)
width = int(nsteps * CTRL_RATE/DURATION)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
for i in range(nu):
  perturb[:, i] = np.convolve(perturb[:, i], kernel, mode='same')

perturbed = np.zeros((10, nu))

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 10
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 180
    step = 0
    while viewer.is_running():
        events = pygame.event.get()
        if not state_handler.paused:
            perturb = np.random.randn(nu)

            mj.mj_step1(model, data)

            lqr_ctrl = get_lqr_ctrl(model, data, K)
            data.ctrl = ctrl0 + lqr_ctrl
            data.ctrl += CTRL_STD*perturb[step]
            ctrl_handler.event_handler(events)
            data.ctrl += ctrl_handler.ctrl

            mj.mj_step2(model, data)

            viewer.sync()
            step += 1
        time.sleep(.007)
        state_handler.event_handler(events)
