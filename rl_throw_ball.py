rerun1 = False
# rerun1 = True
rerun2 = False
# rerun2 = True

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
from matplotlib import pyplot as plt

from rl_utils import *
import gymnasium as gym
import random
import torch
import matplotlib.pyplot as plt

### Set things up
seed = 2
out_f = 'grab_ball_ctrl.npy'

Tk = 120
lr = 1/Tk
# max_its = 400
max_its = 200
# max_its = 120
n_episode = 10000

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
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    body_pos=body_pos,)
model = env.model
data = env.data

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
    ctrls = grab_ball.right_arm_target_traj(
        env, full_traj, targ_traj_mask, targ_traj_mask_type, ctrls, 30, seed,
        CTRL_RATE, CTRL_STD, Tk, lr=lr, max_its=max_its)
    np.save(out_f, ctrls)
else:
    ctrls = np.load(out_f)

# util.reset(model, data, 10, body_pos)
# grab_ball.forward_to_contact(env, ctrls, True)

# ctrls_end = np.tile(ctrls[-1], (200, 1))
# ctrls_end[:, right_arm_with_adh] = 0
# util.reset(model, data, 10, body_pos)
# grab_ball.forward_to_contact(env, np.concatenate([ctrls, ctrls_end]), True)

# sys.exit()

util.reset(model, data, 10, body_pos)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

# total_num_episodes = int(5e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []

# obs = env.reset_model(10)
# obs = wrapped_env.reset_model(seed=seed, n_steps=10)
# obs = wrapped_env.reset(seed=seed, n_steps=10)

for seed in [1]:  # Fibonacci seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if rerun2:
        # Reinitialize agent every seed
        agent = REINFORCE(obs_space_dims, action_space_dims)
        reward_over_episodes = []

        # First train agent to match ctrls:
        ctrlst = torch.tensor(ctrls, dtype=torch.float32)

        # opt = torch.optim.Adam(agent.net.parameters(), lr=.1)
        opt = torch.optim.SGD(agent.net.parameters(), lr=.01)
        losses = []
        tic = time.time()
        first_time = tic
        latest_time = tic

        progress_bar = util.ProgressBar(update_every=2, final_it=n_episode)

        for episode in range(n_episode):
            opt.zero_grad()
            options = dict(render=False, n_steps=10)
            obs = wrapped_env.reset(seed=seed, options=options)
            # obs = util.reset(model, data, 10, body_pos)
            # util.reset(model, data, 9, body_pos)
            # action = data.ctrl.copy()
            # obs, reward, terminated, truncated, info = wrapped_env.step(action)
            # Gradient descent to output current ctrl policy
            loss = 0
            Tk1 = int(Tk / 3)
            for k in range(Tk-1):
                loss_factor = 1
                if k < Tk1:
                    loss_factor = 4
                action = agent.sample_action(obs[0])
                loss += loss_factor*((action - ctrlst[k])**2).mean()
                obs = env.step(action.detach().numpy(), render=False)
            loss /= Tk
            loss.backward()
            opt.step()
            losses.append(loss.item())
            progress_bar.update()
            # print(losses[-1])
        obs = wrapped_env.reset(seed=seed, options=options)
        actions = np.zeros((Tk,action_space_dims))
        torch.manual_seed(seed)
        for k in range(Tk):
            action = agent.sample_action(obs[0]).detach().numpy()
            actions[k] = action
            obs = env.step(action, render=False)

        save_dict = {'state_dict': agent.net.state_dict(), 'losses': losses,
                     'actions': actions}
        torch.save(save_dict, f'net_params_{seed}.pt')
        # plt.plot(losses); plt.show()
        breakpoint()
        optionsn = dict(render=True, n_steps=10)
        wrapped_env.reset(seed=seed, options=optionsn)
        # actions_full = np.concatenate((actions, np.zeros_like(actions)))
        grab_ball.forward_to_contact(env, actions, True)
    else:
        breakpoint()
        options = dict(render=False, n_steps=10)
        agent = REINFORCE(obs_space_dims, action_space_dims)
        obs = wrapped_env.reset(seed=seed, options=options)
        save_dict = torch.load(f'net_params_{seed}.pt')
        agent.net.load_state_dict(save_dict['state_dict'])
        losses = save_dict['losses']
        actions = save_dict['actions']

        # actions_full = np.concatenate((actions, np.zeros_like(actions)))
        # wrapped_env.reset(seed=seed, options=options)
        # grab_ball.forward_to_contact(env, actions_full, True)
    obs = wrapped_env.reset(seed=seed, options=options)
    grab_ball.forward_to_contact(env, actions, True)
    breakpoint()

    ctrls_with_end = np.concatenate([ctrls, ctrls_end])
    Tkf = ctrls_with_end.shape[0]
    noisev = grab_ball.make_noisev(model, seed, Tkf, CTRL_STD, CTRL_RATE)
    options = dict(render=False, n_steps=10)
    obs = wrapped_env.reset(seed=seed, options=options)
    breakpoint()
    grab_ball.forward_to_contact(env, ctrls_with_end, True)
    breakpoint()

    options = dict(render=False, n_steps=10)
    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed, options=options)

        done = False
        for tk in range(Tkf):
            action = agent.sample_action(obs)

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

        breakpoint()
        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)
