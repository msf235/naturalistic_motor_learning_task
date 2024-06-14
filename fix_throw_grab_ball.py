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
import mujoco as mj

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

render_mode = 'human'
keyframe = 'wide'

outdir = Path('output')
outdir.mkdir(parents=True, exist_ok=True)
# savedir = Path('data/phase_1/ball_grab')
# only_save_states = False
savedir = Path('data/phase_2/ball_grab')
only_save_states = True
savedir.mkdir(parents=True, exist_ok=True)

shutil.copy('humanoid.xml', savedir)
shutil.copy('humanoid_and_tennis.xml', savedir)
shutil.copy('tennis_serve_scene.xml', savedir)

for num in range(1, 4):
    ### Set things up
    seed = 2
    out_f = outdir/f'ball_grab_ctrl_seed_{seed}_{num}.pkl'

    env = humanoid2d.Humanoid2dEnv(
        render_mode=render_mode,
        frame_skip=1,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        reset_noise_scale=0,
        xml_file='./humanoid_and_baseball.xml',
        keyframe_name='wide',)
    model = env.model
    data = env.data

    dt = model.opt.timestep
    burn_step = int(.1 / dt)
    # reset = lambda : opt_utils.reset(model, data, burn_step, 2*burn_step, keyframe)
    reset = lambda : opt_utils.reset(model, data, burn_step, 2*burn_step, keyframe)

    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    ctrls_burn_in = load_data['ctrls_burn_in']
    lowest_losses = load_data['lowest_losses']

    # reset()
    ctrls = lowest_losses.peekitem(0)[1][1]
    # arm_t.forward_with_sites(env, ctrls, ['hand_right'], render=True)
    reset()
    qs, vs, ss = util.forward_sim(model, data, ctrls)
    system_states = np.hstack((qs, vs))
    fn = f'ball_grab_{num}_'
    np.save(savedir/ (fn + 'states.npy'), system_states)
    if not only_save_states:
        np.save(savedir/(fn + 'ctrls.npy'), ctrls)
        np.save(savedir/(fn + 'contact.npy'), ss)
    # reset()
    # arm_t.forward_with_sites(env, ctrls_full, sites, render=True)
    # mj.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    # util.forward_sim(model, data, ctrls_full)
