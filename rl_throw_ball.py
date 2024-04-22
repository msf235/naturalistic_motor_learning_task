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
from matplotlib import pyplot as plt

### Set things up
seed = 2
out_f = 'grab_ball_ctrl.npy'
rerun = False
# rerun = True

Tk = 120

body_pos = -0.3

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}
# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(
    render_mode='human',
    # render_mode='rgb_array',
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG)
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
non_right_arm_a = acts['non_right_arm']
all_act_ids = list(range(model.nu))

env.reset(seed=seed) # necessary?
util.reset(model, data, 10, body_pos)

# Get noise
# CTRL_STD = .05       # actuator units
CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds

full_traj = grab_ball.throw_traj(model, data, Tk)

targ_traj_mask = np.ones((Tk-1,))
targ_traj_mask_type = 'progressive'

lr = 2/Tk
max_its = 100

noisev = grab_ball.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

if rerun or not os.path.exists(out_f):
    ### Get initial stabilizing controls
    util.reset(model, data, 10, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['non_adh'],
        joints['body'], free_ctrls=np.ones((Tk,1)))[:2]
    util.reset(model, data, 10, body_pos)
    ctrls = grab_ball.right_arm_target_traj(
        env, full_traj, targ_traj_mask, targ_traj_mask_type, ctrls, 30, seed,
        CTRL_RATE, CTRL_STD, Tk,
        lr=lr, max_its=max_its)
    np.save(out_f, ctrls)
else:
    ctrls = np.load(out_f)

util.reset(model, data, 10, body_pos)
grab_ball.forward_to_contact(env, ctrls, True)
