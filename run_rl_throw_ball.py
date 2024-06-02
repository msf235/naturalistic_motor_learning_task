import humanoid2d as h2d
import opt_utils as opt_utils
import numpy as np
import sim_util as util
import mujoco as mj
import sys
from pathlib import Path
import time
import pickle as pkl
import arm_targ_traj as arm_t
from matplotlib import pyplot as plt

from rl_utils import *
import gymnasium as gym
import random
import torch
import matplotlib.pyplot as plt
import pickle as pkl

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
out_f = outdir / 'grab_ball_ctrl.npy'

Tf = 1.2
# Tf = 2
# Tf = 2.3

# Adam
opt = 'adam'
# lr = .003
lr = .001
# lr = .01
lr2 = .0005
# lr = 1
# lr2 = .5

# lr = 2/Tk
# lr2 = .06
max_its = 1000
# # max_its = 200
# # max_its = 120
# n_episode = 10000

it_lr2 = int(max_its*.8)

# rerun1 = False
rerun1 = True
# rerun2 = False
# rerun2 = True
render_mode = 'human'
# render_mode = 'rgb_array'

keyframe = 'baseball_pos'

# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    keyframe_name=keyframe,)
model = env.model
data = env.data

dt = model.opt.timestep
burn_step = int(.1 / dt)
# reset = lambda : opt_utils.reset(model, data, burn_step, 2*burn_step, keyframe)
reset = lambda : opt_utils.reset(model, data, burn_step, 2*burn_step, keyframe)

reset()

# Tk = 2*120
Tk = int(Tf / dt)

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

right_arm_with_adh = acts['right_arm']
n_adh = len(acts['adh'])

# env.reset(seed=seed) # necessary?
reset()
# util.reset(model, data, 10, body_pos)

# Get noise
# CTRL_STD = .05       # actuator units
CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds

full_traj, time_dict = arm_t.throw_traj(model, data, Tk)

targ_traj_mask = np.ones((Tk,))
targ_traj_mask_type = 'progressive'

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

baseball_idx = arm_t.one_arm_idxs(model)

sites = ['hand_right']
grad_idxs = [baseball_idx['arm_a_without_adh']]
target_trajs = [full_traj]
masks = [targ_traj_mask]
mask_types = [targ_traj_mask_type]

grab_t = Tf / 2.2
grab_tk = int(grab_t/dt)

bodyj = joints['body']['body_dofs']

q_targ = np.zeros((Tk, 2*model.nq))
q_targ_nz = np.linspace(0, -2.44, time_dict['Tk2'])
q_targ[time_dict['t_1']:time_dict['t_2'], 
        joints['all']['wrist_left']] = q_targ_nz
q_targ[time_dict['t_2']:, joints['all']['wrist_left']] = -2.44
q_targs = [q_targ]
q_targ_mask = np.zeros((Tk,2*model.nq))
q_targ_mask[time_dict['t_1']:] = 1
q_targ_masks = [q_targ_mask]
q_targ_mask_types = ['const']

incr_every = 50
t_incr = Tf
amnt_to_incr = int(t_incr/dt)
# t_grad = 0.05
# t_grad = Tf * .04
t_grad = Tf * .1
# grad_update_every = 10
grad_update_every = 1
grad_trunc_tk = int(t_grad/(grad_update_every*dt))
grab_phase_it=15
# grab_phase_it=0

grab_time = time_dict['t_1']
let_go_times = [Tk+1]
contact_check_list = [['ball', 'hand_right1'], ['ball', 'hand_right2']]
adh_ids = [acts['adh_right_hand'][0], acts['adh_right_hand'][0]]
let_go_ids = acts['adh_right_hand']

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    reset()
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        bodyj,
        free_ctrls=np.zeros((Tk,n_adh)))[:2]
    # util.reset(model, data, 10, body_pos)
    # arm_t.forward_to_contact(env, ctrls+noisev, True)
    reset()
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env, sites, grad_idxs, baseball_idx['not_arm_j'],
        baseball_idx['not_arm_a'], target_trajs, masks, mask_types,
        q_targs, q_targ_masks, q_targ_mask_types, ctrls, grad_trunc_tk,
        seed,
        CTRL_RATE, CTRL_STD, Tk, lr=lr, lr2=lr2,
        it_lr2=it_lr2,
        max_its=max_its, keep_top=10,
        incr_every=incr_every, amnt_to_incr=amnt_to_incr,
        grad_update_every=grad_update_every,
        grab_phase_it=grab_phase_it,
        grab_phase_tk=grab_tk,
        phase_2_it=max_its+1,
        optimizer=opt,
        contact_check_list=contact_check_list,
        adh_ids=adh_ids,
        let_go_times=let_go_times,
        let_go_ids=let_go_ids,
        grab_time=grab_time
    )
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses}, f)
    # np.save(out_f, ctrls)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']
ctrls = lowest_losses.peekitem(0)[1][1]
hxs, qs = arm_t.forward_with_sites(env, ctrls, sites, render=False)
# qs_wr = qs[:, joints['all']['wrist_left']]
# q_targs_wr = q_targ[:, joints['all']['wrist_left']]
n = len(sites)
grads = np.nan*np.ones((n,) + ctrls.shape)
tt = np.arange(0, Tf, dt)
fig, axs = plt.subplots(3, n, figsize=(5*n, 5))
if n == 1:
    axs = axs.reshape((3,1))
while True:
    arm_t.show_plot(hxs, target_trajs, masks, None, None, sites, grad_idxs,
                    ctrls, axs, grads, tt)
    # plt.show()
    fig.show()
    plt.pause(1)
    reset()
    hxs, qs = arm_t.forward_with_sites(env, ctrls, sites, render=True)
    # ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
    # reset()




ctrls = lowest_losses.peekitem(0)[1][1]
reset()

dt = model.opt.timestep
T = Tk*dt
tt = np.arange(0, T-dt, dt)
n = len(sites)
fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
for k in range(n):
    hx = arm_t.forward_with_site(env, ctrls+noisev, sites[k], False)
    loss = np.mean((hx - target_trajs[k])**2)
    util.reset(model, data, 10, body_pos)
    ax = axs[k]
    ax.plot(tt, hx[:,1], color='blue', label='x')
    ax.plot(tt, target_trajs[k][:,1], '--', color='blue')
    ax.plot(tt, hx[:,2], color='red', label='y')
    ax.plot(tt, target_trajs[k][:,2], '--', color='red')
    ax.legend()
plt.show()
util.reset(model, data, 10, body_pos)
arm_t.forward_to_contact(env, ctrls+noisev, True)

breakpoint()


if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    util.reset(model, data, 10, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        joints['body']['all'], free_ctrls=np.ones((Tk,n_adh)))[:2]
    # util.reset(model, data, 10, body_pos)
    # arm_t.forward_to_contact(env, ctrls+noisev, True)
    util.reset(model, data, 10, body_pos)
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env, full_traj, targ_traj_mask, targ_traj_mask_type, ctrls, 30, seed,
        CTRL_RATE, CTRL_STD, Tk, lr=lr, max_its=max_its, keep_top=10)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses}, f)
    # np.save(out_f, ctrls)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

ctrls = lowest_losses.peekitem(0)[1][1]
util.reset(model, data, 10, body_pos)
hxs1 = arm_t.forward_with_site(env, ctrls+noisev, 'hand_right', True)
loss1 = np.mean((hxs1 - full_traj)**2)
util.reset(model, data, 10, body_pos)
hxs2 = arm_t.forward_with_site(env, ctrls+noisev, 'base', False)
loss2 = np.mean((hxs2 - full_traj)**2)
print(hxs2[-5:,:5])
print(loss1, loss2)
dt = model.opt.timestep
T = Tk*dt
tt = np.arange(0, T-dt, dt)
fig, ax = plt.subplots()
ax.plot(tt, hxs1[:,1], color='blue', label='x')
ax.plot(tt, hxs2[:,1], '-.', color='blue', label='x_ball')
ax.plot(tt, full_traj[:,1], '--', color='blue')
ax.plot(tt, hxs1[:,2], color='red', label='y')
ax.plot(tt, hxs2[:,2], '-.', color='red', label='y_ball')
ax.plot(tt, full_traj[:,1], '--', color='blue')
ax.plot(tt, full_traj[:,2], '--', color='red')
ax.legend()
plt.show()
breakpoint()

Te = 250
util.reset(model, data, 10, body_pos)
ctrls_end = np.zeros((Te, model.nu))
ctrls_best = lowest_losses.peekitem(0)[1][1]
arm_t.forward_to_contact(env, np.concatenate([ctrls_best, ctrls_end]), True)
breakpoint()

ctrls_best = lowest_losses.peekitem(0)[1][1]

util.reset(model, data, 10, body_pos)
ctrls_end = np.zeros((Te, model.nu))
ctrls_with_end_best = np.concatenate([ctrls_best, ctrls_end])
util.reset(model, data, 10, body_pos)
arm_t.forward_to_contact(env, ctrls_with_end_best, True)

breakpoint()

ctrls_end = np.tile(ctrls[-1], (149, 1))
ctrls_end[:, right_arm_with_adh] = 0
ctrls_with_end = np.concatenate([ctrls, ctrls_end])
Tkf = ctrls_with_end.shape[0] + 1

# util.reset(model, data, 10, body_pos)
# arm_t.forward_to_contact(env, ctrls_with_end, True)

util.reset(model, data, 10, body_pos)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e3)  # Total number of episodes
obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []

render = dict(render=True, n_steps=10)
no_render = dict(render=False, n_steps=10)

for seed in [1]:  # Fibonacci seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if rerun2:
        # Reinitialize agent every seed
        agent = REINFORCE(obs_space_dims, action_space_dims)

        # First train agent to match ctrls:
        # ctrlst = torch.tensor(ctrls, dtype=torch.float32)
        ctrlst = torch.tensor(ctrls_with_end, dtype=torch.float32)

        # opt = torch.optim.Adam(agent.net.parameters(), lr=.1)
        opt = torch.optim.SGD(agent.net.parameters(), lr=lr2)
        losses = []
        tic = time.time()
        first_time = tic
        latest_time = tic

        progress_bar = util.ProgressBar(update_every=2, final_it=n_episode)

        for episode in range(n_episode):
            opt.zero_grad()
            agent.clear_episode()
            obs = wrapped_env.reset(seed=seed, options=no_render)
            loss = 0
            Tk1 = int(Tk / 3)
            for k in range(Tkf-1):
                loss_factor = 1
                if k < Tk1:
                    loss_factor = 4
                # if Tk-1 <= tk and tk <= Tk+1:
                    # loss_factor = 4
                # Should probs be reset every episode?
                action = agent.sample_action(obs[0])
                loss += loss_factor*((action - ctrlst[k])**2).mean()
                obs = env.step(action.detach().numpy(), render=False)
            loss /= Tkf
            loss.backward()
            opt.step()
            losses.append(loss.item())
            progress_bar.update()
            # print(losses[-1])
        obs = wrapped_env.reset(seed=seed, options=no_render)
        actions = np.zeros((Tkf-1, action_space_dims))
        torch.manual_seed(seed)
        for k in range(Tkf-1):
            action = agent.sample_action(obs[0]).detach().numpy()
            actions[k] = action
            obs = env.step(action, render=False)

        save_dict = {'state_dict': agent.net.state_dict(), 'losses': losses,
                     'actions': actions}
        torch.save(save_dict, f'net_params_{seed}.pt')
    else:
        agent = REINFORCE(obs_space_dims, action_space_dims)
        obs = wrapped_env.reset(seed=seed, options=no_render)
        save_dict = torch.load(f'net_params_{seed}.pt')
        agent.net.load_state_dict(save_dict['state_dict'])
        losses = save_dict['losses']
        actions = save_dict['actions']

    # obs = wrapped_env.reset(seed=seed, options=no_render)
    # arm_t.forward_to_contact(env, actions, True)

    progress_bar = util.ProgressBar(update_every=2,
                                    final_it=total_num_episodes)

    reward_over_episodes = []
    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed, options=no_render)

        actions = np.zeros((Tkf-1, action_space_dims))
        done = False
        for tk in range(Tkf-1):
            action = agent.sample_action(obs)
            actions[tk] = action.detach().numpy()

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(
                action.detach().numpy())
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated
            if done:
                break
        progress_bar.update()

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 100 == 0:
            avg_reward = np.round(np.mean(wrapped_env.return_queue), 3)
            print()
            print("Episode:", episode, "Average Reward:", avg_reward)
            print()
            # obs, info = wrapped_env.reset(seed=seed, options=no_render)
            # arm_t.forward_to_contact(env, actions, True)

    rewards_over_seeds.append(reward_over_episodes)

    breakpoint()
