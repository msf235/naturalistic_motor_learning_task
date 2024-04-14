import humanoid2d as h2d
# import baseball_lqr as lqr
import opt_utils as opt_utils
import numpy as np
import sim_util as util
import mujoco as mj
import sys
import os
import copy
import time
import pickle as pkl
import grab_ball

### Set things up
seed = 2
out_f = 'grab_ball_ctrl.np'
# rerun = False
rerun = True

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
not_right_arm_j = [i for i in body_j if i not in right_arm_j]
acts = opt_utils.get_act_names(model)
right_arm_a = acts['right_arm']
adh = acts['adh_right_hand']
non_adh = acts['non_adh']
other_a = acts['non_right_arm']
all_act_ids = list(range(model.nu))

env.reset(seed=seed) # necessary?

util.reset(model, data, 10, body_pos)

# show_forward_sim(model, data, np.zeros((Tk, model.nu)))

# Get noise
CTRL_STD = .05       # actuator units
# CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds

# if rerun or not os.path.exists(out_f):

if rerun or not os.path.exists(out_f):
    target = data.site('target')
    util.reset(model, data, 10, body_pos)
    ctrls, k = grab_ball.right_arm_target(env, target, body_pos,
                                          seed, CTRL_RATE, CTRL_STD, Tk)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'k': k}, f)
else:
    with open(out_f, 'rb') as f:
        out = pkl.load(f)
        ctrls = out['ctrls']
        k = out['k']

noisev = grab_ball.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

util.reset(model, data, 10, body_pos)
grab_ball.show_forward_sim(env, (ctrls+noisev)[:k+1])

sys.exit()

# ctrl = ctrls[k]
# ctrl[adh] = 1
# ctrl[right_arm_a] = -.1
fact = -.7
throw_ctrls = np.zeros((Tk-1, 2))
throw_ctrls[:, 0] = fact
throw_ctrls[:, 1] = fact
qpos0 = data.qpos.copy()
ctrls2 = opt_utils.get_stabilized_ctrls(
    model, data, Tk, noisev, qpos0, other_a, not_right_arm_j,
    fact*np.ones((Tk-1, 2))
)[0]
ctrls2[:,adh]=1
util.reset(model, data, 10, body_pos)
grab_ball.show_forward_sim(env, ctrls+noisev)
grab_ball.show_forward_sim(env, ctrls2)

throw_targ = [-0.2, 0.4, 0]

# breakpoint()
