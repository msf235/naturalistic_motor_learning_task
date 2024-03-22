import humanoid2d as h2d
# import baseball_lqr as lqr
import opt_utils as opt_utils
import numpy as np
import sim_utils as util
import mujoco as mj
import sys

### Set things up
seed = 2
rng = np.random.default_rng(seed)

Tk = 50

body_pos = -0.4

# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(
    render_mode='human',
    # render_mode='rgb_array',
    frame_skip=1)
env.reset(seed=seed)
model = env.model
data = env.data

joints = opt_utils.get_joint_names(model)
acts = opt_utils.get_act_names(model)
right_arm_j = joints['right_arm_joint_inds']
# right_arm_a = joints['right_arm_act_inds']
# other_a = joints['non_right_arm_act_inds']
adh = acts['adh_right_hand']

def show_forward_sim(model, data, ctrls):
    for k in range(ctrls.shape[0]-1):
        env.step(ctrls[k])

# Get noise
CTRL_STD = .05       # actuator units
# CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds
width = int(CTRL_RATE/model.opt.timestep)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
noise = util.FilteredNoise(model.nu, kernel, rng)
noisev = CTRL_STD * noise.sample(Tk-1)
noisev[:, adh] = 0

### Get initial stabilizing controls
util.reset(model, data, 10, body_pos)
ctrls, K = opt_utils.get_stabilized_ctrls(model, data, Tk, noisev,
                                          data.qpos.copy(), free_act_ids=adh)[:2]
util.reset(model, data, 10, body_pos)

show_forward_sim(model, data, ctrls+noisev)

qs, qvels = util.forward_sim(model, data, ctrls)

util.reset(model, data, 10, body_pos)

### Gradient descent

qpos0 = data.qpos.copy()

def get_losses(model, data, site1, site2):
    # I could put a forward sim here for safety (but less efficient)
    dlds = site1.xpos - site2.xpos
    C = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)
    dldq = C.T @ dlds
    lams_fin = dldq
    return np.zeros(Tk), lams_fin


for k0 in range(3):
    lr = 20
    lams_fin = get_losses(model, data, data.site('hand_right'),
                          data.site('target'))[1]
    util.reset(model, data, 10, body_pos)
    show_forward_sim(model, data, ctrls+noisev)
    util.reset(model, data, 10, body_pos)
    grads = opt_utils.traj_deriv(model, data, qs, qvels, ctrls, lams_fin,
                                 np.zeros(Tk), fixed_act_inds=other_a)
    ctrls[:,right_arm_a] = ctrls[:, right_arm_a] - lr*grads[:Tk-1]

    qs, qvels = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, qpos0, 10, right_arm_a, ctrls[:, right_arm_a]
    )[2:]

print(qs[-3:,:3])


