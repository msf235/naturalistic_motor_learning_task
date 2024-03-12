import mujoco as mj
import time
import sys
import numpy as np
import scipy
import control_logic as cl
import humanoid2d as h2d
import baseball_lqr as lqr
import base

# xml_file = 'humanoid_and_baseball.xml'
# with open(xml_file, 'r') as f:
  # xml = f.read()

env = h2m.Humanoid2dEnv(render_mode='human')

model = env.model
data = env.data

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

# K = get_feedback_ctrl_matrix(model, data, Q, R)

ctrl_handler = cl.modelControlHandler(model, data, gain=.1)

data.ctrl = lqr_ctrl.ctrl0

K = get_feedback_ctrl_matrix(model, data, Q, R)
# K = get_feedback_ctrl_matrix(model, data, Q, R)

CTRL_STD = 0.05       # actuator units
CTRL_RATE = 0.8       # seconds
# Precompute some noise.
np.random.seed(1)
width = int(CTRL_RATE/model.opt.timestep)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)

state_handler = cl.simulationStateHandler()
state_handler.paused = True
state_handler.paused = False
perturb = np.random.randn(nu, len(kernel))

with mj.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 10
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 180
    while viewer.is_running():
        events = pygame.event.get()
        if not state_handler.paused:
            # No need to flip kernel since symmetric
            perturb_smoothed = perturb @ kernel
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
        time.sleep(.007)
        state_handler.event_handler(events)
