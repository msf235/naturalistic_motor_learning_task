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


Tk = 120*2
# Tk = 120*2
# Tk = 320
# lr = 1/Tk
# lr = 10/Tk
lr = 2/Tk
max_its = 600
# max_its = 200
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
full_traj2[:] = data.site('hand_left').xpos.copy()
targ_traj_mask2 = np.ones((Tk-1,))
targ_traj_mask_type2 = 'progressive'

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

lr = .3/Tk

out_f = Path(str(out_f_base) + '_both.pkl')

bodyj = joints['body']['body_dofs']

sites = ['hand_right', 'hand_left', 'racket_handle']
full_trajs = [full_traj1, full_traj2, full_traj1]
masks = [targ_traj_mask1, targ_traj_mask2, targ_traj_mask1]
mask_types = [targ_traj_mask_type1, targ_traj_mask_type2, targ_traj_mask_type1]

tennis_idxs = arm_t.tennis_idxs(model)
site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                  tennis_idxs['left_arm_without_adh'],
                  tennis_idxs['right_arm_without_adh']]
stabilize_jnt_idx = tennis_idxs['not_arm_j']
stabilize_act_idx = tennis_idxs['not_arm_a']

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
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env, sites, site_grad_idxs, stabilize_jnt_idx, stabilize_act_idx,
        full_trajs, masks, mask_types, ctrls, 30, seed, CTRL_RATE, CTRL_STD,
        Tk, lr=lr, max_its=max_its, keep_top=10, incr_per1=5, incr_per2=5)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses}, f)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

n = len(sites)
nr = range(n)
ctrls = lowest_losses.peekitem(0)[1][1]
util.reset(model, data, 10, body_pos)
util.reset(model, data, 10, body_pos)
dt = model.opt.timestep
T = Tk*dt
tt = np.arange(0, T-dt, dt)

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
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

breakpoint()

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
    ctrls, lowest_losses = arm_t.two_arm_target_traj_tennis(env,
        full_traj1, targ_traj_mask1, targ_traj_mask_type1,
        full_traj2, targ_traj_mask2, targ_traj_mask_type2,
        ctrls, 30, seed,
        CTRL_RATE, CTRL_STD, Tk, lr=lr, max_its=max_its, keep_top=10,
        incr_per1=5, incr_per2=5)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses}, f)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

ctrls = lowest_losses.peekitem(0)[1][1]
util.reset(model, data, 10, body_pos)
hxs1 = arm_t.forward_with_site(env, ctrls+noisev, 'hand_right', False)
loss1 = np.mean((hxs1 - full_traj1)**2)
util.reset(model, data, 10, body_pos)
hxs2 = arm_t.forward_with_site(env, ctrls+noisev, 'racket_handle', False)
loss2 = np.mean((hxs2 - full_traj1)**2)
print(hxs2[-5:,:5])
print(loss1, loss2)
dt = model.opt.timestep
T = Tk*dt
tt = np.arange(0, T-dt, dt)

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
target_traj = full_traj1 * targ_traj_mask1.reshape(-1, 1)
ax = axs[0]
ax.plot(tt, hxs1[:,1], color='blue', label='x')
ax.plot(tt, full_traj1[:,1], '--', color='blue')
ax.plot(tt, hxs1[:,2], color='red', label='y')
ax.plot(tt, target_traj[:,2], '--', color='red')
ax.set_title('Right hand')
ax.legend()
# ax = axs[1]
# target_traj = target_traj2 * targ_traj_mask2.reshape(-1, 1)
# ax.plot(tt, hxs2[:,1], color='blue', label='x')
# ax.plot(tt, target_traj[:,1], '--', color='blue')
# ax.plot(tt, hxs2[:,2], color='red', label='y')
# ax.plot(tt, target_traj[:,2], '--', color='red')
# ax.set_title('Left hand')
# ax.legend()
# ax = axs[2]
ax = axs[1]
target_traj = full_traj1 * targ_traj_mask1.reshape(-1, 1)
ax.plot(tt, hxs2[:,1], color='blue', label='x')
ax.plot(tt, target_traj[:,1], '--', color='blue')
ax.plot(tt, hxs2[:,2], color='red', label='y')
ax.plot(tt, target_traj[:,2], '--', color='red')
ax.set_title('Tennis handle')
ax.legend()
fig.tight_layout()
plt.show()

util.reset(model, data, 10, body_pos)
arm_t.forward_to_contact(env, ctrls, True)

breakpoint()


