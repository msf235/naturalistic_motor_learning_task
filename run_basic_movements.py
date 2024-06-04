import humanoid2d
import opt_utils
import numpy as np
import sim_util as util
import mujoco as mj
import sys
from pathlib import Path
import pickle as pkl
import arm_targ_traj as arm_t
import basic_movements as bm
from matplotlib import pyplot as plt
import gymnasium as gym
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
seed = 1
out_f_base = outdir/f'basic_movement_seed_{seed}'

Tk = 1200
# Tk = 320
lr = 1/Tk
# max_its = 400
max_its = 200
# max_its = 120

CTRL_STD = 0
CTRL_RATE = 1

# rerun1 = False
rerun1 = True

# render_mode = 'human'
render_mode = 'rgb_array'

# body_pos = -0.3
body_pos = 0

# Create a Humanoid2dEnv object
env = humanoid2d.Humanoid2dEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    # xml_file='./humanoid.xml',
    body_pos=body_pos,)
model = env.model
data = env.data

burn_steps = 100
dt = model.opt.timestep

num = seed

util.reset(model, data, burn_steps, body_pos)

np.random.seed(seed)
torch.manual_seed(seed)

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

smoothing_sigma = int(.1 / dt)
arc_std = 0.2
# Move right arm while keeping left arm fixed
rs, thetas = bm.random_arcs_right_arm(model, data, Tk-1,
                                      data.site('hand_right').xpos,
                                      smoothing_sigma, arc_std)
traj1_xs = np.zeros((Tk-1, 3))
traj1_xs[:,1] = rs * np.cos(thetas)
traj1_xs[:,2] = rs * np.sin(thetas)
traj1_xs += data.site('shoulder1_right').xpos
full_traj = traj1_xs
targ_traj_mask = np.ones((Tk-1,))
targ_traj_mask_type = 'progressive'

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)


out_f = Path(str(out_f_base) + f'_right_{num}.pkl')

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    util.reset(model, data, burn_steps, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        joints['body']['body_dofs'], free_ctrls=np.ones((Tk,len(acts['adh'])))
    )[:2]
    util.reset(model, data, burn_steps, body_pos)
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env, full_traj, targ_traj_mask, targ_traj_mask_type, ctrls, 30, seed,
        CTRL_RATE, CTRL_STD, Tk, lr=lr, max_its=max_its, keep_top=10,
        right_or_left='right')
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses,
                  'ctrls_burn_in': np.zeros((burn_steps, model.nu))
                 }, f)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

tt = np.arange(0, (Tk-1)*model.opt.timestep, model.opt.timestep)
ctrls = lowest_losses.peekitem(0)[1][1]
util.reset(model, data, burn_steps, body_pos)
hxs1 = arm_t.forward_with_site(env, ctrls, 'hand_right', True)
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig, ax = plt.subplots()
target_traj = full_traj * targ_traj_mask.reshape(-1, 1)
# ax = axs[0]
ax.plot(tt, hxs1[:,1], color='blue', label='x')
ax.plot(tt, target_traj[:,1], '--', color='blue')
ax.plot(tt, hxs1[:,2], color='red', label='y')
ax.plot(tt, target_traj[:,2], '--', color='red')
ax.set_title('Right hand')
ax.legend()
plt.show()
util.reset(model, data, burn_steps, body_pos)
arm_t.forward_to_contact(env, ctrls, True)

util.reset(model, data, burn_steps, body_pos)


## Move left arm while keeping right arm fixed
rs, thetas = bm.random_arcs_left_arm(model, data, Tk-1,
                                      data.site('hand_left').xpos,
                                     smoothing_sigma, .02)
traj1_xs = np.zeros((Tk-1, 3))
traj1_xs[:,1] = rs * np.cos(thetas)
traj1_xs[:,2] = rs * np.sin(thetas)
traj1_xs += data.site('shoulder1_left').xpos
full_traj = traj1_xs
targ_traj_mask = np.ones((Tk-1,))
targ_traj_mask_type = 'progressive'

# print(data.site('hand_left').xpos, full_traj[0])
# print()

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

out_f = Path(str(out_f_base) + f'_left_{num}.pkl')

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    util.reset(model, data, burn_steps, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        joints['body']['body_dofs'], free_ctrls=np.ones((Tk,1)))[:2]
    util.reset(model, data, burn_steps, body_pos)
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env, full_traj, targ_traj_mask, targ_traj_mask_type, ctrls, 30, seed,
        CTRL_RATE, CTRL_STD, Tk, lr=lr, max_its=max_its, keep_top=10,
        right_or_left='left')
    util.reset(model, data, burn_steps, body_pos)
    # arm_t.forward_to_contact(env, ctrls_best, True)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses,
                  'ctrls_burn_in': np.zeros((burn_steps, model.nu))
                 }, f)
    # qs, vs = util.forward_sim(model, data, ctrls)
    # system_states = np.hstack((qs, vs))
    # ctrls_best = lowest_losses.peekitem(0)[1][1]
    # np.save(str(out_f_base) + '_left_ctrls.npy', ctrls_best)
    # np.save(str(out_f_base) + '_left_states.npy', system_states)
    # np.save(out_f, ctrls)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

util.reset(model, data, burn_steps, body_pos)
hxs1 = arm_t.forward_with_site(env, ctrls, 'hand_left', True)
ctrls = lowest_losses.peekitem(0)[1][1]
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig, ax = plt.subplots()
target_traj = full_traj * targ_traj_mask.reshape(-1, 1)
# ax = axs[0]
ax.plot(tt, hxs1[:,1], color='blue', label='x')
ax.plot(tt, target_traj[:,1], '--', color='blue')
ax.plot(tt, hxs1[:,2], color='red', label='y')
ax.plot(tt, target_traj[:,2], '--', color='red')
ax.set_title('Right hand')
ax.legend()
plt.show()

util.reset(model, data, burn_steps, body_pos)
arm_t.forward_to_contact(env, ctrls, True)

util.reset(model, data, burn_steps, body_pos)
# print(data.site('hand_left').xpos, full_traj[0])


## Move both arms simultaneously
rs, thetas = bm.random_arcs_right_arm(model, data, Tk-1,
                                      data.site('hand_right').xpos,
                                      smoothing_sigma, .02)
traj1_xs = np.zeros((Tk-1, 3))
traj1_xs[:,1] = rs * np.cos(thetas)
traj1_xs[:,2] = rs * np.sin(thetas)
traj1_xs += data.site('shoulder1_right').xpos
full_traj1 = traj1_xs
targ_traj_mask1 = np.ones((Tk-1,))
targ_traj_mask_type1 = 'progressive'

rs, thetas = bm.random_arcs_left_arm(model, data, Tk-1,
                                      data.site('hand_left').xpos,
                                     smoothing_sigma, .02)
traj2_xs = np.zeros((Tk-1, 3))
traj2_xs[:,1] = rs * np.cos(thetas)
traj2_xs[:,2] = rs * np.sin(thetas)
traj2_xs += data.site('shoulder1_left').xpos
full_traj2 = traj2_xs
targ_traj_mask2 = np.ones((Tk-1,))
targ_traj_mask_type2 = 'progressive'

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

lr = .1/Tk

out_f = Path(str(out_f_base) + f'_both_{num}.pkl')

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    util.reset(model, data, burn_steps, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        joints['body']['body_dofs'], free_ctrls=np.ones((Tk,1)))[:2]
    util.reset(model, data, burn_steps, body_pos)
    ctrls, lowest_losses = arm_t.two_arm_target_traj(
        env,
        full_traj1, targ_traj_mask1, targ_traj_mask_type1,
        full_traj2, targ_traj_mask2, targ_traj_mask_type2,
        ctrls, 30, seed,
        CTRL_RATE, CTRL_STD, Tk, lr=lr, max_its=max_its, keep_top=10)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses,
                  'ctrls_burn_in': np.zeros((burn_steps, model.nu))
                 }, f)
    # qs, vs = util.forward_sim(model, data, ctrls)
    # system_states = np.hstack((qs, vs))
    # ctrls_best = lowest_losses.peekitem(0)[1][1]
    # np.save(str(out_f_base) + f'_both_ctrls_{num}.npy', ctrls_best)
    # np.save(str(out_f_base) + f'_both_states_{num}.npy', system_states)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

ctrls_best = lowest_losses.peekitem(0)[1][1]
util.reset(model, data, burn_steps, body_pos)
arm_t.forward_to_contact(env, ctrls_best, True)

