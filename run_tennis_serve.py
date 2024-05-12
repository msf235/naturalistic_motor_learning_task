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
out_f = outdir/'tennis_ctrl.pkl'


# Tk = 120
Tk = 120*8
# Tk = 320
# lr = 1/Tk
# lr = 10/Tk
lr = 2/Tk
max_its = 1200
# max_its = 200
# max_its = 120

CTRL_STD = 0
CTRL_RATE = 1

# rerun1 = False
rerun1 = True

render_mode = 'human'
# render_mode = 'rgb_array'

body_pos = -0.4

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

targ_traj_mask1 = np.ones((Tk,))
targ_traj_mask_type1 = 'progressive'

full_traj1, full_traj2 = arm_t.tennis_traj(model, data, Tk)
print(data.site('hand_left').xpos)
targ_traj_mask2 = np.ones((Tk,))
targ_traj_mask_type2 = 'progressive'

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

lr = .3/Tk

bodyj = joints['body']['body_dofs']

sites = ['hand_right', 'hand_left', 'racket_handle']
full_trajs = [full_traj1, full_traj2, full_traj1]
masks = [targ_traj_mask1, targ_traj_mask2, targ_traj_mask1]
mask_types = [targ_traj_mask_type1, targ_traj_mask_type2, targ_traj_mask_type1]

tennis_idxs = arm_t.two_arm_idxs(model)
site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                  tennis_idxs['left_arm_without_adh'],
                  tennis_idxs['right_arm_without_adh']]
stabilize_jnt_idx = tennis_idxs['not_arm_j']
stabilize_act_idx = tennis_idxs['not_arm_a']

n = len(sites)
nr = range(n)

dt = model.opt.timestep
T = Tk*dt
tt = np.arange(0, T-dt, dt)

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    util.reset(model, data, 10, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        bodyj, free_ctrls=np.ones((Tk, len(acts['adh'])))
    )[:2]
    # util.reset(model, data, 10, body_pos)
    # arm_t.forward_to_contact(env, ctrls, True)
    util.reset(model, data, 10, body_pos)
    print(data.site('hand_left').xpos)
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env, sites, site_grad_idxs, stabilize_jnt_idx, stabilize_act_idx,
        full_trajs, masks, mask_types, ctrls, 30, seed, CTRL_RATE, CTRL_STD,
        Tk, lr=lr, max_its=max_its, keep_top=10, incr_every=20,
        amnt_to_incr=80)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses}, f)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

# ctrls = lowest_losses.peekitem(0)[1][1]
util.reset(model, data, 10, body_pos)

fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
for k in nr:
    hx = arm_t.forward_with_site(env, ctrls, sites[k], False)
    loss = np.mean((hx - full_trajs[k])**2)
    ax = axs[k]
    ax.plot(tt, hx[:,1], color='blue', label='x')
    ax.plot(tt, full_trajs[k][:,1], '--', color='blue')
    ax.plot(tt, hx[:,2], color='red', label='y')
    ax.plot(tt, full_trajs[k][:,2], '--', color='red')
    ax.set_title(sites[k])
    ax.legend()
fig.tight_layout()
plt.show()

util.reset(model, data, 10, body_pos)
arm_t.forward_to_contact(env, ctrls, True)

