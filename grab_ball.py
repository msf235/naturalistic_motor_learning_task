import humanoid2d as h2d
# import baseball_lqr as lqr
import opt_utils as opt_utils
import numpy as np
import sim_utils as util
import mujoco as mj
import sys
import os
import pickle as pkl

### Set things up
seed = 2
rng = np.random.default_rng(seed)
out_f = 'grab_ball_ctrl.np'
rerun = False
# rerun = True

# Tk = 500
Tk = 50

body_pos = -0.4

# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(
    render_mode='human',
    # render_mode='rgb_array',
    frame_skip=1)
model = env.model
data = env.data

joints = opt_utils.get_joint_names(model)
right_arm_j = joints['right_arm']
body_j = joints['body']
acts = opt_utils.get_act_names(model)
right_arm_a = acts['right_arm']
adh = acts['adh_right_hand']
other_a = acts['non_right_arm']
all_act_ids = list(range(model.nu))

env.reset(seed=seed) # necessary?

def show_forward_sim(model, data, ctrls):
    for k in range(ctrls.shape[0]-1):
        # if k == 30:
            # breakpoint()
        env.step(ctrls[k])

util.reset(model, data, 10, body_pos)

# show_forward_sim(model, data, np.zeros((Tk, model.nu)))

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


def get_losses(model, data, site1, site2):
    # I could put a forward sim here for safety (but less efficient)
    dlds = site1.xpos - site2.xpos
    C = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)
    dldq = C.T @ dlds
    lams_fin = dldq
    return np.zeros(Tk), lams_fin

if rerun or not os.path.exists(out_f):
    ### Get initial stabilizing controls
    util.reset(model, data, 10, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), all_act_ids, body_j)[:2]
    util.reset(model, data, 10, body_pos)

    show_forward_sim(model, data, ctrls+noisev)

    qs, qvels = util.forward_sim(model, data, ctrls)

    util.reset(model, data, 10, body_pos)

    ### Gradient descent
    qpos0 = data.qpos.copy()

    ball_contact = False
    for k0 in range(10):
        lr = 10
        lams_fin = get_losses(model, data, data.site('hand_right'),
                              data.site('target'))[1]
        util.reset(model, data, 10, body_pos)
        if k0 > 1:
            for k in range(Tk-1):
                util.step(model, data, ctrls[k]+noisev[k])
                env.render()
                contact_pairs = util.get_contact_pairs(model, data)
                for cp in contact_pairs:
                    if 'ball' in cp and 'hand_right' in cp:
                        ball_contact = True
                        break
        else:
            show_forward_sim(model, data, ctrls+noisev)
        if ball_contact:
            break
        grads = opt_utils.traj_deriv(model, data, qs, qvels, ctrls, lams_fin,
                                     np.zeros(Tk), fixed_act_inds=other_a)
        ctrls[:, right_arm_a] = ctrls[:, right_arm_a] - lr*grads[:Tk-1]
        qs, qvels = opt_utils.get_stabilized_ctrls(
            model, data, Tk, noisev, qpos0, other_a, right_arm_a,
            ctrls[:, right_arm_a]
        )[2:]

    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'k': k}, f)
else:
    with open(out_f, 'rb') as f:
        out = pkl.load(f)
        ctrls = out['ctrls']
        k = out['k']

util.reset(model, data, 10, body_pos)
show_forward_sim(model, data, ctrls+noisev)

ctrl = ctrls[k]
ctrl[adh] = 1
ctrl[right_arm_a] = -.2
# data.ctrl[-1] = 1
# data.ctrl[5] = 1
for k in range(100):
    util.step(model, data, ctrl)
    env.render()

