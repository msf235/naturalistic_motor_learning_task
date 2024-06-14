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

outdir = Path('output')
savedir = Path('data/phase_5')
outdir.mkdir(parents=True, exist_ok=True)
savedir.mkdir(parents=True, exist_ok=True)

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
    keyframe_name=keyframe,)
model = env.model
data = env.data

shutil.copy('humanoid.xml', savedir)
shutil.copy('humanoid_and_tennis.xml', savedir)
shutil.copy('tennis_serve_scene.xml', savedir)

num = 1

for num in range(1, 4):
    out_f = outdir/f'tennis_ctrl_working{num}.pkl'
    # save_str = str(savedir/f'tennis_working_{num}')
    dt = model.opt.timestep
    Ta = 2
    Tk = int(Ta / dt)

    joints = opt_utils.get_joint_ids(model)
    acts = opt_utils.get_act_ids(model)
    out = arm_t.tennis_traj(model, data, Tk)
    right_hand_traj, left_hand_traj, ball_traj, time_dict = out

    burn_step = int(.1 / dt)
    # reset = lambda : opt_utils.reset(model, data, burn_step, 2*burn_step, keyframe)
    reset = lambda : opt_utils.reset(model, data, burn_step, 2*burn_step, keyframe)

    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    lowest_losses = load_data['lowest_losses']

    ctrls = lowest_losses.peekitem(0)[1][1]
    # ctrls[time_dict['t_left_1']:, 0] = .4
    ctrls = np.vstack((ctrls, np.zeros((Tk, model.nu))))

    # sites = ['hand_right', 'hand_left']
    sites = ['hand_right']
    # reset()
    # hxs, qs = arm_t.forward_with_sites(env, ctrls, sites, render=False)
    qs, vs = util.forward_sim(model, data, ctrls)
    system_states = np.hstack((qs, vs))
    fn = f'tennis_serve_{num}_'
    np.save(savedir/(fn + 'ctrls.npy'), ctrls)
    np.save(savedir/(fn + 'states.npy'), system_states)
    # np.save(save_str + '_sensors.npy', )
    # tt = np.arange(0, hxs[0].shape[0], dt)
    # qs_wr = qs[:, joints['all']['wrist_left']]
    # q_targs_wr = q_targ[:, joints['all']['wrist_left']]
    # grads = np.nan*np.ones((len(sites),) + ctrls.shape)
    # fig, axs = plt.subplots(3, n, figsize=(5*n, 5))
    # while True:
        # # arm_t.show_plot(hxs, full_trajs, masks,
                        # # # qs_wr, q_targs_wr,
                        # # sites, site_grad_idxs, ctrls, axs,
                        # # grads, tt)
        # # # plt.show()
        # # fig.show()
        # # plt.pause(1)
        # reset()
        # arm_t.forward_with_sites(env, ctrls, sites, render=True)
        # print('waiting')
        # time.sleep(20)
