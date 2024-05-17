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
Tk = 120*5
# Tk = 120*6
# Tk = 320
# lr = .5/Tk
# lr = .05/Tk
lr = 10/Tk
# lr = 2/Tk
# max_its = 1200*3
# max_its = 2000
max_its = 1000
# max_its = 1200
# max_its = 1600
# max_its = 200
# max_its = 120

CTRL_STD = 0
CTRL_RATE = 1

# rerun1 = False
rerun1 = True

render_mode = 'human'
# render_mode = 'rgb_array'

keyframe = 'wide_tennis_pos'
# keyframe = 'test'

reset = lambda : opt_utils.reset(model, data, 40, 30, keyframe)

# Create a Humanoid2dEnv object
env = humanoid2d.Humanoid2dEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    xml_file='./humanoid_and_tennis.xml',
    keyframe_name='wide_tennis_pos',)
model = env.model
data = env.data

reset()
# tmp = util.get_contact_pairs(model, data)

# targ_traj_mask = np.ones((Tk,))
# targ_traj_mask[Tk//3:] = 0
targ_traj_mask = np.zeros((Tk,))
# targ_traj_mask = np.ones((Tk,))
# targ_traj_mask[4*40] = 1
targ_traj_mask[:4*40+1] = 1
# targ_traj_mask_type = 'progressive'
targ_traj_mask_type = 'const'

out = arm_t.tennis_traj(model, data, Tk)
right_hand_traj, left_hand_traj, ball_traj, time_dict = out

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

bodyj = joints['body']['body_dofs']

sites = ['hand_right', 'hand_left', 'racket_handle', 'ball']
sites = ['hand_right']
full_trajs = [right_hand_traj, left_hand_traj, right_hand_traj, ball_traj]
full_trajs = [right_hand_traj]
# masks = [targ_traj_mask]*4
masks = [targ_traj_mask]
# mask_types = [targ_traj_mask_type]*4
mask_types = [targ_traj_mask_type]

tennis_idxs = arm_t.two_arm_idxs(model)
# site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                  # tennis_idxs['left_arm_without_adh'],
                  # tennis_idxs['right_arm_without_adh'],
                  # tennis_idxs['left_arm_without_adh']]
site_grad_idxs = [tennis_idxs['right_arm_without_adh']]
stabilize_jnt_idx = tennis_idxs['not_arm_j']
stabilize_act_idx = tennis_idxs['not_arm_a']

n = len(sites)
nr = range(n)

dt = model.opt.timestep
T = Tk*dt
tt = np.arange(0, T, dt)
left_adh_act_vals = np.ones((Tk-1, 1))
left_adh_act_vals[time_dict['t_left_3']:] = 0

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    reset()
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        bodyj, free_ctrls=np.ones((Tk, len(acts['adh'])))
    )[:2]
    ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
    # reset()
    # arm_t.forward_to_contact(env, ctrls, True)
    reset()
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env, sites, site_grad_idxs, stabilize_jnt_idx, stabilize_act_idx,
        full_trajs, masks, mask_types, ctrls, 30, seed, CTRL_RATE, CTRL_STD,
        Tk, lr=lr, max_its=max_its, keep_top=10, incr_every=80,
        amnt_to_incr=50, grad_update_every=1)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses}, f)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

ctrls = lowest_losses.peekitem(0)[1][1]
# ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
# reset()
# arm_t.forward_to_contact(env, ctrls, True)
reset()

fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
if n == 1:
    axs = [axs]
for k in nr:
    hx = arm_t.forward_with_site(env, ctrls, sites[k], False)
    dlds = hx - full_trajs[k]
    loss = np.mean(dlds**2)
    reset()
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

reset()
arm_t.forward_to_contact(env, ctrls, True)
breakpoint()

ctrls_end = np.zeros((100, model.nu))
ctrls_end[:] = ctrls[-1]
# ctrls_end[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals[-1]
ctrls = np.concatenate((ctrls, ctrls_end), axis=0)
Tk = ctrls.shape[0] + 1
T = Tk*dt
tt = np.arange(0, T, dt)
reset()
arm_t.forward_to_contact(env, ctrls, True)

