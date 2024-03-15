import humanoid2d as h2d
import baseball_lqr as lqr
import numpy as np
import sim_utils as util
import mujoco as mj
import sys

seed = 2

# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(render_mode='human', frame_skip=1)
env.reset(seed=seed)
model = env.model
data = env.data

# Get noise
# CTRL_STD = 0.05       # actuator units
CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds
width = int(CTRL_RATE/model.opt.timestep)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
noise = util.FilteredNoise(model.nu, kernel, 3*seed+7)

# Get initial stabilizing controls
## Reset and burn in:
def reset(model, data, nsteps):
    mj.mj_resetData(model, data)
    for k in range(nsteps):
        mj.mj_step(model, data)

reset(model, data, 10)

qpos0 = data.qpos.copy()
ctrl0 = lqr.get_ctrl0(model, data)
data.ctrl = ctrl0
rv = np.ones(model.nu)
K = lqr.get_feedback_ctrl_matrix(model, data)

Tk = 300
qs = np.zeros((Tk, model.nq)); vs = np.zeros((Tk, model.nq))
qs[0] = qpos0; vs[0] = data.qvel.copy()
us = np.zeros((Tk, model.nu))
losses = np.zeros((Tk,))
ctrls = np.zeros((Tk-1, model.nu))

data.ctrl[:] = ctrl0
ctrl = ctrl0

for k in range(Tk-1):
    ctrl = lqr.get_lqr_ctrl_from_K(model, data, K, qpos0n, ctrl0)
    ctrls[k] = ctrl
    out = env.step(ctrl + CTRL_STD*noise.sample())
    observation, reward, terminated, __, info = out
    qs[k+1] = observation[:model.nq]
    vs[k+1] = observation[model.nq:]


# Gradient descent

sites1 = data.site('hand_right').xpos
sites2 = data.site('target').xpos
dlds = sites1 - sites2
# curr_loss = .5*np.linalg.norm(dlds)**2
# Cs = np.zeros((Tk, 3, model.nv))
C = np.zeros((3, model.nv))
mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)

dldq = C.T @ dlds
lams_fin = dldq

joints = lqr.get_joint_names(model)
right_arm_j = joints['right_arm_joint_inds']

while True:
    sites1 = data.site('hand_right').xpos
    sites2 = data.site('target').xpos
    dlds = sites1 - sites2
    C = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)

    dldq = C.T @ dlds
    lams_fin = dldq

    reset(model, data, 10)
    grads = util.traj_deriv(model, data, qs, vs, us, lams_fin, losses,
                            fixed_act_inds=fixed_act_inds)
    ctrls[:,free_act_inds] = ctrls[:, free_act_inds] - .01*grads[:Tk-1]

    # ctrls_lqr = np.zeros((Tk-1, model.nu))

    # for k in range(Tk-1):
        # ctrl = lqr.get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0)
        # ctrls[k] = ctrl
        # out = env.step(ctrl + CTRL_STD*noise.sample())
        # observation, reward, terminated, __, info = out
        # qs[k+1] = observation[:model.nq]
        # vs[k+1] = observation[model.nq:]

    # mj.mj_resetData(model, data)
    # data.ctrl = ctrl0
    # ctrl = lqr.get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0)
    # ctrls[k] = ctrl
    K = lqr.get_feedback_ctrl_matrix(model, data, excluded_state_inds, rv)

    env.reset(seed=seed)
    reset(model, data, 10)
    for k in range(Tk-1):
        # ctrl_lqr = lqr.get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0)
        # ctrl = ctrls[k]
        # if k % 10 == 0:
        # if k > 20 == 0:
        if k > - 1 == 0:
            # ctrl0 = lqr.get_ctrl0(model, data)
            # # qpos0 = data.qpos.copy()
            # data.ctrl[:] = ctrl0
            # K = lqr.get_feedback_ctrl_matrix(model, data, excluded_state_inds, rv)
            ctrl_lqr = lqr.get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0)
            # ctrl[included_act_inds] = ctrl_lqr[included_act_inds]
            ctrl = ctrl_lqr
        out = env.step(ctrl + CTRL_STD*noise.sample())
        observation, reward, terminated, __, info = out
        qs[k+1] = observation[:model.nq]
        vs[k+1] = observation[model.nq:]


