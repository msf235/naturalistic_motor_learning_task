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

Tk = 120
lr = 1/Tk
lr2 = .06
# max_its = 400
max_its = 200
# max_its = 120
n_episode = 10000

rerun1 = False
# rerun1 = True
rerun2 = False
# rerun2 = True
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

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

right_arm_with_adh = acts['right_arm']
n_adh = len(acts['adh'])

env.reset(seed=seed) # necessary?
util.reset(model, data, 10, body_pos)

# Get noise
# CTRL_STD = .05       # actuator units
CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds

full_traj = arm_t.throw_traj(model, data, Tk)

targ_traj_mask = np.ones((Tk-1,))
targ_traj_mask_type = 'progressive'

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    util.reset(model, data, 10, body_pos)
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        joints['body']['all'], free_ctrls=np.ones((Tk,n_adh)))[:2]
    util.reset(model, data, 10, body_pos)
    arm_t.forward_to_contact(env, ctrls+noisev, True)
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


# util.reset(model, data, 10, body_pos)
# arm_t.forward_to_contact(env, ctrls, True)
# breakpoint()

ctrls_best = lowest_losses.peekitem(0)[1][1]

arm_a = acts['right_arm']
not_arm_a = [k for k in acts['all'] if k not in arm_a and k not in acts['adh']]
# Remove any remaining adhesion actuators
not_arm_j = [i for i in joints['body']['body_dofs'] if i not in
             joints['body']['right_arm'] and i not in joints['body']['right_arm']]
arm_a_without_adh = [k for k in arm_a if k not in acts['adh']]
# Include all adhesion (including other hand)
arm_with_all_adh = [k for k in acts['all'] if k in arm_a or k in acts['adh']]

util.reset(model, data, 10, body_pos)
arm_t.forward_to_contact(env, ctrls_best, False)
# ctrls_end_best = np.tile(ctrls_best[-1], (149, 1))
# ctrls_end_best[:, right_arm_with_adh] = 0
# ctrls_with_end_best = np.concatenate([ctrls_best, ctrls_end_best])
Te = 250
# util.reset(model, data, 10, body_pos)
# ctrls_end = np.zeros((Te, model.nu))
noisev = arm_t.make_noisev(model, seed, Te, CTRL_STD, CTRL_RATE)
ctrls_end, __, qs, qvels = opt_utils.get_stabilized_ctrls(
    model, data, Te, noisev, data.qpos.copy(),
    acts['not_adh'], not_arm_j, np.zeros((Te, n_adh)),
)
ctrls_with_end_best = np.concatenate([ctrls_best, ctrls_end])
# ctrls, K = opt_utils.get_stabilized_ctrls(
    # model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
    # joints['body']['all'], free_ctrls=np.ones((Tk,n_adh)))[:2]
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
