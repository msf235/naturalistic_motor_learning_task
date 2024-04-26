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
import gym
import random
import torch
import matplotlib.pyplot as plt

### Set things up
seed = 2
out_f = 'grab_ball_ctrl.npy'
# rerun = False
rerun = True

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
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    body_pos=body_pos,)
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

# util.reset(model, data, 10, body_pos)
# ctrls_full = np.concatenate((ctrls, np.zeros_like(ctrls)))
# grab_ball.forward_to_contact(env, ctrls, True)
# sys.exit()

util.reset(model, data, 10, body_pos)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e3)  # Total number of episodes
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
    if rerun:
        # Reinitialize agent every seed
        agent = REINFORCE(obs_space_dims, action_space_dims)
        reward_over_episodes = []

        # First train agent to match ctrls:
        ctrlst = torch.tensor(ctrls, dtype=torch.float32)

        # opt = torch.optim.Adam(agent.net.parameters(), lr=.1)
        opt = torch.optim.SGD(agent.net.parameters(), lr=.01)
        losses = []
        for episode in range(10000):
            opt.zero_grad()
            options = dict(render=False)
            obs = wrapped_env.reset(seed=seed, n_steps=10, options=options)
            # obs = util.reset(model, data, 10, body_pos)
            # util.reset(model, data, 9, body_pos)
            # action = data.ctrl.copy()
            # obs, reward, terminated, truncated, info = wrapped_env.step(action)
            # Gradient descent to output current ctrl policy
            loss = 0
            Tk1 = int(Tk / 3)
            for k in range(Tk):
                loss_factor = 1
                if k < Tk1:
                    loss_factor = 5
                action = agent.sample_action(obs[0])
                loss += loss_factor*((action - ctrlst[k])**2).mean()
                obs = env.step(action.detach().numpy(), render=False)
            loss /= Tk
            loss.backward()
            opt.step()
            losses.append(loss.item())
            print(losses[-1])
        obs = wrapped_env.reset(seed=seed, n_steps=10, options=options)
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
        wrapped_env.reset(seed=seed, n_steps=10)
        # actions_full = np.concatenate((actions, np.zeros_like(actions)))
        grab_ball.forward_to_contact(env, actions, True)
    else:
        options = dict(render=False)
        agent = REINFORCE(obs_space_dims, action_space_dims)
        obs = wrapped_env.reset(seed=seed, n_steps=10, options=options)
        save_dict = torch.load(f'net_params_{seed}.pt')
        agent.net.load_state_dict(save_dict['state_dict'])
        losses = save_dict['losses']

        actions = np.zeros((Tk,action_space_dims))

        torch.manual_seed(seed)
        for k in range(Tk):
            action = agent.sample_action(obs[0])
            actions[k] = action.detach().numpy()
            obs = env.step(actions[k], render=False)

        actions2 = save_dict['actions']

        actions_full = np.concatenate((actions, np.zeros_like(actions)))
        wrapped_env.reset(seed=seed, n_steps=10)
        grab_ball.forward_to_contact(env, actions, True)

    breakpoint()

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)
