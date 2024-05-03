import humanoid2d
import opt_utils
import numpy as np
import sim_util as util
import sys
from pathlib import Path
import pickle as pkl
import arm_targ_traj as arm_t
import basic_movements as bm
from matplotlib import pyplot as plt
import torch

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

outdir = Path('output')
outdir.mkdir(parents=True, exist_ok=True)

### Set things up
seed = 2
out_f_base = outdir/'move_right_arm_ctrl'

Tk = 120
# Tk = 320
lr = 1/Tk
# max_its = 400
max_its = 200
# max_its = 120

CTRL_STD = 0
CTRL_RATE = 1

# rerun1 = False
rerun1 = True

render_mode = 'human'
# render_mode = 'rgb_array'

body_pos = -0.4
# body_pos = 0

# Create a Humanoid2dEnv object
env = humanoid2d.Humanoid2dEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    xml_file='./humanoid_and_tennis.xml',
    body_pos=body_pos,)
model = env.model
data = env.data

util.reset(model, data, 10, body_pos)

full_traj1 = arm_t.tennis_traj(model, data, Tk)

targ_traj_mask1 = np.ones((Tk-1,))
targ_traj_mask_type1 = 'progressive'

# Move both arms simultaneously
full_traj2 = np.zeros(full_traj1.shape)
targ_traj_mask2 = np.ones((Tk-1,))
targ_traj_mask_type2 = 'progressive'

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

lr = .1/Tk

out_f = Path(str(out_f_base) + '_both.pkl')

bodyj = joints['body']['body_dofs']

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    util.reset(model, data, 10, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        bodyj, free_ctrls=np.ones((Tk,1)))[:2]
    util.reset(model, data, 10, body_pos)
    arm_t.forward_to_contact(env, ctrls, True)
    util.reset(model, data, 10, body_pos)
    ctrls, lowest_losses = arm_t.two_arm_target_traj(env,
        full_traj1, targ_traj_mask1, targ_traj_mask_type1,
        full_traj2, targ_traj_mask2, targ_traj_mask_type2,
        ctrls, 30, seed,
        CTRL_RATE, CTRL_STD, Tk, lr=lr, max_its=max_its, keep_top=10)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses}, f)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

ctrls_best = lowest_losses.peekitem(0)[1][1]
util.reset(model, data, 10, body_pos)
arm_t.forward_to_contact(env, ctrls_best, True)

