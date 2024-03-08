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
# def conv_term(a, v, k):
    # if len(v) > len(a):
        # a, v = v, a
    # vf = np.flip(v)
    # n = a.shape[0]
    # m = v.shape[0]
    # ret_val = vf[max(0, m-k-1):min(m, n-k+m-1)] @ a[max(0, k-m+1):min(n, k+1)]
    # return ret_val

# a = np.random.randn(100)
# v = np.random.randn(10)
# n3 = min(len(a), len(v))
# n4 = max(len(a), len(v))
# k1 = (n3+1) // 2 - 1
# k2 = n4 + k1 - 1
# res11 = conv_term(a, v, k1)
# res12 = conv_term(a, v, k2)
# resf2 = np.convolve(a, v, mode='same')
# res21 = resf2[0]
# res22 = resf2[-1]
# breakpoint()

# perturb_org = perturb.copy()
# for i in range(nu):
  # perturb[:, i] = np.convolve(perturb[:, i], kernel, mode='same')
# temp = np.convolve(perturb_org[:,0], kernel, mode='same')

size = 10
perturbed = np.zeros((size, nu))

state_handler = cl.simulationStateHandler()
state_handler.paused = True
state_handler.paused = False
perturb = np.random.randn(nu, len(kernel))
perturb_rolled = np.zeros((nu, len(kernel)))
perturb_smoothed_list = []

# for k in range(50):
    # perturb_smoothed = perturb @ kernel
    # perturb_smoothed_list.append(perturb_smoothed)
    # perturb = np.roll(perturb, -1, axis=1)
    # perturb[:, -1] = np.random.randn(nu)

# from matplotlib import pyplot as plt
# plt.plot(perturb_smoothed_list)
# plt.show()
# breakpoint()

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 10
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 180
    # step = 0
    while viewer.is_running():
        events = pygame.event.get()
        if not state_handler.paused:
            # No need to flip kernel since symmetric
            perturb_smoothed = perturb @ kernel
            perturb_smoothed_list.append(perturb_smoothed)
            perturb[:] = np.roll(perturb, -1, axis=1)
            perturb[:, -1] = np.random.randn(nu)

            mj.mj_step1(model, data)

            lqr_ctrl = get_lqr_ctrl(model, data, K)
            data.ctrl = ctrl0 + lqr_ctrl
            data.ctrl += CTRL_STD*perturb_smoothed
            ctrl_handler.event_handler(events)
            data.ctrl += ctrl_handler.ctrl

            mj.mj_step2(model, data)

            viewer.sync()
            # step += 1
        time.sleep(.007)
        state_handler.event_handler(events)
