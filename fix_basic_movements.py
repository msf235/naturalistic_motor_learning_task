
import humanoid2d
import opt_utils
import numpy as np
import sim_util as util
import mujoco as mj
import sys
import shutil
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
# savedir = Path('data/phase_1/basic')
# only_save_state = False
savedir = Path('data/phase_2/basic')
only_save_state = True
savedir.mkdir(parents=True, exist_ok=True)

shutil.copy('humanoid.xml', savedir)
shutil.copy('humanoid_and_basic.xml', savedir)
shutil.copy('basic_scene.xml', savedir)

### Set things up
# seed = 2

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

for num in range(1, 4):
    np.random.seed(num)
    torch.manual_seed(num)
    out_f_base = outdir/f'basic_movement_seed_{num}'

    out_f = Path(str(out_f_base) + f'_right_{num}.pkl')
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    lowest_losses = load_data['lowest_losses']
    ctrls = lowest_losses.peekitem(0)[1][1]

    util.reset(model, data, burn_steps, body_pos)
    qs, vs, ss = util.forward_sim(model, data, ctrls)
    system_states = np.hstack((qs, vs))
    ctrls_best = lowest_losses.peekitem(0)[1][1]
    fn = f'basic_movement_right_{num}_'
    np.save(str(savedir) + '/' + fn + 'states.npy', system_states)
    if not only_save_state:
        np.save(str(savedir) + '/' + fn + 'ctrls.npy', ctrls_best)
        np.save(str(savedir) + '/' + fn + 'contact.npy', ss)

    out_f = Path(str(out_f_base) + f'_left_{num}.pkl')
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    lowest_losses = load_data['lowest_losses']
    ctrls = lowest_losses.peekitem(0)[1][1]

    util.reset(model, data, burn_steps, body_pos)
    qs, vs, ss = util.forward_sim(model, data, ctrls)
    system_states = np.hstack((qs, vs))
    ctrls_best = lowest_losses.peekitem(0)[1][1]
    fn = f'basic_movement_left_{num}_'
    np.save(str(savedir) + '/' + fn + 'states.npy', system_states)
    if not only_save_state:
        np.save(str(savedir) + '/' + fn + 'ctrls.npy', ctrls_best)
        np.save(str(savedir) + '/' + fn + 'contact.npy', ss)

    out_f = Path(str(out_f_base) + f'_both_{num}.pkl')
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    lowest_losses = load_data['lowest_losses']
    ctrls = lowest_losses.peekitem(0)[1][1]

    util.reset(model, data, burn_steps, body_pos)
    qs, vs, ss = util.forward_sim(model, data, ctrls)
    system_states = np.hstack((qs, vs))
    ctrls_best = lowest_losses.peekitem(0)[1][1]
    fn = f'basic_movement_both_{num}_'
    np.save(str(savedir) + '/' + fn + 'states.npy', system_states)
    if not only_save_state:
        np.save(str(savedir) + '/' + fn + 'ctrls.npy', ctrls_best)
        np.save(str(savedir) + '/' + fn + 'contact.npy', ss)
