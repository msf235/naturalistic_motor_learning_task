import humanoid2d as h2d
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
import basic_movements as bm
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

### Set things up
seed = 2
out_f = 'move_right_arm_ctrl.npy'

Tk = 120
lr = 2/Tk
lr2 = .06
# max_its = 400
max_its = 200
# max_its = 120

CTRL_STD = 0
CTRL_RATE = 1

# rerun1 = False
rerun1 = True

render_mode = 'human'
# render_mode = 'rgb_array'

body_pos = -0.3

# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    body_pos=body_pos,)
model = env.model
data = env.data


util.reset(model, data, 10, body_pos)
# Move right arm while keeping left arm fixed
rs, thetas = bm.random_arcs_right_arm(model, data, Tk-1)
traj1_xs = np.zeros((Tk-1, 3))
traj1_xs[:,1] = rs * np.cos(thetas)
traj1_xs[:,2] = rs * np.sin(thetas)
# traj1_xs[:,1] = rs * np.cos(thetas[0])
# traj1_xs[:,2] = rs * np.sin(thetas[0])
traj1_xs += data.site('shoulder1_right').xpos
full_traj = traj1_xs

targ_traj_mask = np.ones((Tk-1,))
targ_traj_mask_type = 'progressive'

noisev = grab_ball.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

joints = opt_utils.get_joint_names(model)
acts = opt_utils.get_act_names(model)

if rerun1 or not os.path.exists(out_f):
    ### Get initial stabilizing controls
    util.reset(model, data, 10, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['non_adh'],
        joints['body'], free_ctrls=np.ones((Tk,1)))[:2]
    util.reset(model, data, 10, body_pos)
    ctrls, lowest_losses = grab_ball.right_arm_target_traj(
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

ctrls_best = lowest_losses.peekitem(0)[1][1]
util.reset(model, data, 10, body_pos)
grab_ball.forward_to_contact(env, ctrls_best, True)

breakpoint()


# Move left arm while keeping right arm fixed
rs, thetas = bm.random_arcs_right_arm(model, data, Tk)
thetas = thetas - np.pi
traj2_xs = np.zeros((Tk, 3))
traj2_xs[:,1] = rs * np.cos(thetas)
traj2_xs[:,2] = rs * np.sin(thetas)
traj2_xs += data.site('shoulder1_left').xpos

# Move both arms simultaneously

joints = opt_utils.get_joint_names(model)

acts = opt_utils.get_act_names(model)
right_arm_with_adh = acts['right_arm_with_adh']

env.reset(seed=seed) # necessary?
util.reset(model, data, 10, body_pos)

# Get noise
# CTRL_STD = .05       # actuator units
CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds

full_traj = grab_ball.throw_traj(model, data, Tk)

targ_traj_mask = np.ones((Tk-1,))
targ_traj_mask_type = 'progressive'

noisev = grab_ball.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

if rerun1 or not os.path.exists(out_f):
    ### Get initial stabilizing controls
    util.reset(model, data, 10, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['non_adh'],
        joints['body'], free_ctrls=np.ones((Tk,1)))[:2]
    util.reset(model, data, 10, body_pos)
    ctrls, lowest_losses = grab_ball.right_arm_target_traj(
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

ctrls_best = lowest_losses.peekitem(0)[1][1]
util.reset(model, data, 10, body_pos)
# grab_ball.forward_to_contact(env, ctrls, True)
# grab_ball.forward_to_contact(env, ctrls_best, True)

ctrls_end_best = np.tile(ctrls_best[-1], (149, 1))
ctrls_end_best[:, right_arm_with_adh] = 0
ctrls_with_end_best = np.concatenate([ctrls_best, ctrls_end_best])
Tkf = ctrls_with_end_best.shape[0] + 1
util.reset(model, data, 10, body_pos)
grab_ball.forward_to_contact(env, ctrls_with_end_best, True)

breakpoint()

ctrls_end = np.tile(ctrls[-1], (149, 1))
ctrls_end[:, right_arm_with_adh] = 0
ctrls_with_end = np.concatenate([ctrls, ctrls_end])
Tkf = ctrls_with_end.shape[0] + 1

# util.reset(model, data, 10, body_pos)
# grab_ball.forward_to_contact(env, ctrls_with_end, True)

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
    # grab_ball.forward_to_contact(env, actions, True)

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
            # grab_ball.forward_to_contact(env, actions, True)

    rewards_over_seeds.append(reward_over_episodes)

    breakpoint()
