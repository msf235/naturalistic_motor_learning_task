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
# Tk = 120*5
# Tk = 120*3
# Tk = 120*4
# Tk = 120*3
# Tk = 120*2
# Tk = 120*6
# Tk = 320
# lr = 1/Tk
# lr = 10/Tk
# lr = .3/Tk
# lr = .03/Tk
# lr = 2/Tk
# lr = 20/Tk
# max_its = 1200*3
max_its = 1500
# max_its = 500
# max_its = 1200
# lr = .0001
# lr = .0005
# lr = .0001
# lr = 5
# lr = 2
# lr = 1
# lr = .01
# lr = .001
# lr = .005
# lr = .0005
# lr = .2
# max_its = 1600
# max_its = 200
# max_its = 120

# Tf = 1.6
Tf = 2.3


CTRL_STD = 0
CTRL_RATE = 1

rerun1 = False
# rerun1 = True

render_mode = 'human'
# render_mode = 'rgb_array'

keyframe = 'wide_tennis_pos'


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

dt = model.opt.timestep

Tk = int(Tf / dt)

# lr = .5/Tk

# Adam
lr = .002
# lr = .005

# SGD
# lr = .5

# SGD with momentum
# lr = .05
# lr = .01
lr = .005

lr2 = .001
it_lr2 = int(max_its*.8)


burn_step = int(.1 / dt)

# reset = lambda : opt_utils.reset(model, data, burn_step, 2*burn_step, keyframe)
reset = lambda : opt_utils.reset(model, data, burn_step, 2*burn_step, keyframe)

reset()
# tmp = util.get_contact_pairs(model, data)

targ_traj_mask = np.ones((Tk,))
# targ_traj_mask_type = 'progressive'
targ_traj_mask_type = 'double_sided_progressive'
# targ_traj_mask_type = 'const'
out = arm_t.tennis_traj(model, data, Tk)
right_hand_traj, left_hand_traj, ball_traj, time_dict = out

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

# lr = .3/Tk

bodyj = joints['body']['body_dofs']

sites = ['hand_right', 'hand_left', 'racket_handle_top', 'ball']
full_trajs = [right_hand_traj, left_hand_traj, right_hand_traj, ball_traj]
masks = [targ_traj_mask]*4
mask_types = [targ_traj_mask_type]*4

tennis_idxs = arm_t.two_arm_idxs(model)
site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                  tennis_idxs['left_arm_without_adh'],
                  tennis_idxs['right_arm_without_adh'],
                  tennis_idxs['left_arm_without_adh']]
stabilize_jnt_idx = tennis_idxs['not_arm_j']
stabilize_act_idx = tennis_idxs['not_arm_a']

n = len(sites)
nr = range(n)

tt = np.arange(0, Tf, dt)
# left_adh_act_vals = np.ones((Tk-1, 1))
# left_adh_act_vals[time_dict['t_left_3']:] = 0

incr_every = 10
# incr_every = 20
# incr_every = 30
grab_t = 1
grab_tk = int(grab_t/dt)
t_incr = 0.08
t_incr = 0.2
amnt_to_incr = int(t_incr/dt)
# t_grad = 0.05
t_grad = 0.1
n_grad = int(t_grad/dt)
grab_phase_it=20

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    reset()
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        bodyj,
        # free_ctrls=np.ones((Tk, len(acts['adh'])))
        free_ctrls=np.zeros((Tk, len(acts['adh'])))
    )[:2]
    # ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
    reset()
    # arm_t.forward_to_contact(env, ctrls, True)
    # reset()
    # breakpoint()
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env, sites, site_grad_idxs, stabilize_jnt_idx, stabilize_act_idx,
        full_trajs, masks, mask_types, ctrls, n_grad, seed, CTRL_RATE, CTRL_STD,
        Tk, lr=lr, lr2=lr2, it_lr2=it_lr2, max_its=max_its, keep_top=10,
        incr_every=incr_every, amnt_to_incr=amnt_to_incr,
        # grad_update_every=10,
        grad_update_every=1, # Need to check this with new code
        grab_phase_it=grab_phase_it,
        grab_phase_tk=grab_tk,
        phase_2_it=Tk+1)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses}, f)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

ctrls = lowest_losses.peekitem(0)[1][1]
reset()
arm_t.forward_with_site(env, ctrls, 'hand_right', render=True)
# ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
reset()

fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
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


