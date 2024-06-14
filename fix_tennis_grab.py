
import humanoid2d
import opt_utils
import numpy as np
import sim_util as util
import sys
import shutil
from pathlib import Path
import pickle as pkl
import arm_targ_traj as arm_t
import basic_movements as bm
from matplotlib import pyplot as plt
import torch
import time

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

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

outdir = Path('output')
outdir.mkdir(parents=True, exist_ok=True)
# savedir = Path('data/phase_1/tennis_grab')
# save_only_state = False
savedir = Path('data/phase_2/tennis_grab')
save_only_state = True
savedir.mkdir(parents=True, exist_ok=True)

shutil.copy('humanoid.xml', savedir/'humanoid_tennis.xml')
shutil.copy('humanoid_and_tennis.xml', savedir)
shutil.copy('tennis_serve_scene.xml', savedir)

### Set things up
num = 3
for num in range(1, 4):
    seed = 2
    out_f = outdir/f'tennis_ctrl_grab_{num}.pkl'

    max_its = 1000
    ctrls = []
    lowest_losses = []

    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
        # ctrls = data['ctrls']
        lowest_losses = load_data['lowest_losses']

    ctrls = lowest_losses.peekitem(0)[1][1]
    qs, vs, ss = util.forward_sim(model, data, ctrls)
    system_states = np.hstack((qs, vs))
    fn = f'tennis_grab_{num}'
    np.save(savedir / (fn + '_states.npy'), system_states)
    if save_only_state:
        np.save(savedir / (fn + '_ctrls.npy'), ctrls)
        np.save(savedir / (fn + '_contact.npy'), ss)
