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
import mujoco as mj
import time

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

outdir = Path('output')
outdir.mkdir(parents=True, exist_ok=True)
savedir = Path('data')
savedir.mkdir(parents=True, exist_ok=True)

### Set things up
seed = 4
out_f = outdir/f'ball_grab_ctrl_seed_{seed}_1.pkl'

max_its = 300

Tf = .9

CTRL_STD = 0
CTRL_RATE = 1

rerun1 = False
# rerun1 = True

render_mode = 'human'
# render_mode = 'rgb_array'

keyframe = 'wide'

# Create a Humanoid2dEnv object
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

ctrls_burn_in = reset()

Tk = int(Tf / dt)

# lr = .5/Tk

# Adam
opt = 'adam'
lr = .001
lr2 = .0005
# lr = 1
# lr2 = .5

# SGD
# opt = 'sgd'
# lr = .01
# lr2 = .005

# SGD with momentum
# lr = .05
# lr = .01
# lr = .0005
# lr = .001
# lr = .002
# lr2 = .0005

it_lr2 = int(max_its*.8)

grab_t = Tf / 2.2
grab_tk = int(grab_t/dt)


targ_traj_mask = np.ones((Tk,))
# targ_traj_mask_type = 'progressive'
targ_traj_mask_type = 'double_sided_progressive'
# targ_traj_mask_type = 'const'
# out = arm_t.tennis_traj(model, data, Tk)
out = arm_t.throw_grab_traj(model, data, Tk)
# right_hand_traj, left_hand_traj, ball_traj, time_dict = out
full_traj, time_dict = out
# ball_traj_mask = np.ones((Tk,))
# ball_traj_mask[time_dict['t_3']:] = 0

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

bodyj = joints['body']['body_dofs']
# bodyj.sort()
# bodyj = joints['body']['all']

# sites = ['hand_right', 'hand_left', 'racket_handle_top', 'ball']
# sites = ['hand_right', 'hand_left', 'racket_handle_top']
sites = ['hand_right']
# full_trajs = [right_hand_traj, left_hand_traj, right_hand_traj, ball_traj]
# full_trajs = [right_hand_traj, left_hand_traj, right_hand_traj]
full_trajs = [full_traj]
# masks = [targ_traj_mask, targ_traj_mask, targ_traj_mask, ball_traj_mask]
# masks = [targ_traj_mask, targ_traj_mask, targ_traj_mask]
masks = [targ_traj_mask]
# mask_types = [targ_traj_mask_type]*4
# mask_types = [targ_traj_mask_type]*3
mask_types = [targ_traj_mask_type]

# tennis_idxs = arm_t.two_arm_idxs(model)
# site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                  # tennis_idxs['left_arm_without_adh'],
                  # tennis_idxs['right_arm_without_adh'],
                  # tennis_idxs['left_arm_without_adh']]
# site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                  # tennis_idxs['left_arm_without_adh'],
                  # tennis_idxs['right_arm_without_adh']]
# stabilize_jnt_idx = tennis_idxs['not_arm_j']
# stabilize_act_idx = tennis_idxs['not_arm_a']

throw_idxs = arm_t.one_arm_idxs(model)
site_grad_idxs = [throw_idxs['arm_a_without_adh']]
stabilize_jnt_idx = throw_idxs['not_arm_j']
stabilize_act_idx = throw_idxs['not_arm_a']

# tmp = util.get_contact_pairs(model, data)

q_targ = np.zeros((Tk, 2*model.nq))
# q_targs = [np.zeros((Tk, 2*model.nq))]*4
# q_targ_masks = [np.zeros((Tk, 2*model.nq))]*4
q_targ_mask = np.zeros((Tk,2*model.nq))
# arm_vels = [model.nv + x for x in joints['body']['right_arm']]
# p1_dur_k = int(.1 / dt)
# q_targ_mask[grab_tk-p1_dur_k:grab_tk, arm_vels] = 1
q_targ_mask2 = np.zeros((Tk,2*model.nq))
q_targ_mask2[time_dict['t_1']:,
            joints['all']['wrist_left']] = 1
q_targ_nz = np.linspace(0, -2.44, time_dict['t_2']-time_dict['t_1'])
q_targ[time_dict['t_1']:time_dict['t_2'], 
        joints['all']['wrist_left']] = q_targ_nz
q_targ[time_dict['t_2']:, joints['all']['wrist_left']] = -2.44
q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask, q_targ_mask]
# q_targ_mask_types = ['const']*4
# q_targ_mask_types = ['const']*3
q_targ_mask_types = ['const']
# q_targs = [q_targ]*4
# q_targs = [q_targ]*3
q_targs = [q_targ]

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

# lr = .3/Tk

n = len(sites)
nr = range(n)

tt = np.arange(0, Tf, dt)
# left_adh_act_vals = np.ones((Tk-1, 1))
# left_adh_act_vals[time_dict['t_left_3']:] = 0

incr_every = 50
# incr_every = 20
# incr_every = 30
# grab_t = Tf / 2
# t_incr = 0.08
# t_incr = Tf * .08
t_incr = Tf
amnt_to_incr = int(t_incr/dt)
# t_grad = 0.05
# t_grad = Tf * .04
t_grad = Tf * .1 #TODO: check if this is better in run_tennis_serve.py
# grad_update_every = 10
grad_update_every = 10
grad_trunc_tk = int(t_grad/(grad_update_every*dt))
grab_phase_it=20
# grab_phase_it=0

# contact_check_list = [['racket_handle', 'hand_right1'], ['racket_handle', 'hand_right2'],
                      # ['ball', 'hand_left1'], ['ball', 'hand_left2']]
contact_check_list = [['ball', 'hand_right1'], ['ball', 'hand_right2']]
# adh_ids = [acts['adh_right_hand'][0], acts['adh_right_hand'][0],
           # acts['adh_left_hand'][0], acts['adh_left_hand'][0]]
adh_ids = [acts['adh_right_hand'][0], acts['adh_right_hand'][0]]
# let_go_ids = [acts['adh_left_hand'][0]]
# let_go_ids = [acts['adh_right_hand'][0]]
let_go_ids = []
# let_go_times = [Tk]
let_go_times = []

# grab_time = int(max(time_dict['t_1']*.9, time_dict['t_left_1']) * .9)
grab_time = int(time_dict['t_1'] * .9)

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    reset()
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        bodyj,
        # free_ctrls=np.ones((Tk, len(acts['adh'])))
        free_ctrls=np.zeros((Tk, len(acts['adh']))),
        let_go_times=let_go_times,
        let_go_ids=let_go_ids,
        contact_check_list=contact_check_list,
        adh_ids=adh_ids,
    )[:2]
    # ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
    reset()
    # arm_t.forward_to_contact(env, ctrls, render=True)
    # reset()
    # breakpoint()
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env, sites, site_grad_idxs, stabilize_jnt_idx, stabilize_act_idx,
        full_trajs, masks, mask_types, q_targs, q_targ_masks,
        q_targ_mask_types, ctrls, grad_trunc_tk, seed, CTRL_RATE, CTRL_STD,
        Tk, lr=lr, lr2=lr2, it_lr2=it_lr2, max_its=max_its, keep_top=10,
        incr_every=incr_every, amnt_to_incr=amnt_to_incr,
        # grad_update_every=10,
        grad_update_every=grad_update_every, # Need to check this with new code
        grab_phase_it=grab_phase_it,
        grab_phase_tk=grab_tk,
        phase_2_it=max_its+1, optimizer=opt,
        contact_check_list=contact_check_list,
        adh_ids=adh_ids,
        let_go_times=let_go_times,
        let_go_ids=let_go_ids,
        grab_time=grab_time
    )
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses,
                  'ctrls_burn_in': ctrls_burn_in}, f)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    # ctrls = load_data['ctrls']
    ctrls_burn_in = load_data['ctrls_burn_in']
    lowest_losses = load_data['lowest_losses']

ctrls = lowest_losses.peekitem(0)[1][1]
# Te = 3
# Tke = int(3 / dt)
# ctrls_end = np.zeros((Tke, model.nu))
# ctrls_full = np.vstack((ctrls, ctrls_end))
# mj.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
# util.forward_sim(model, data, ctrls_burn_in)
# reset()
# for tk in range(ctrls_burn_in.shape[0]):
    # mj
# hxs, qs = arm_t.forward_with_sites(env, ctrls, sites, render=False)
# # qs_wr = qs[:, joints['all']['wrist_left']]
# qs_wr = qs[:, joints['all']['wrist_right']]
# # q_targs_wr = q_targ[:, joints['all']['wrist_left']]
# q_targs_wr = q_targ[:, joints['all']['wrist_right']]
# grads = np.nan*np.ones((len(sites),) + ctrls.shape)
# fig, axs = plt.subplots(3, n, figsize=(5*n, 5))
# axs = axs.reshape((3, 1))
while True:
    # arm_t.show_plot(hxs, full_trajs, masks,
                    # # qs_wr, q_targs_wr,
                    # sites, site_grad_idxs, ctrls, axs,
                    # grads, tt)
    # # plt.show()
    # fig.show()
    # plt.pause(1)
    # mj.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    # util.forward_sim(model, data, ctrls_burn_in)
    reset()
    arm_t.forward_with_sites(env, ctrls, sites, render=True)
    time.sleep(5)
    # ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
    # reset()

# for k in nr:

    # hx = hxs[k]
    # dlds = hx - full_trajs[k]
    # loss = np.mean(dlds**2)
    # reset()
    # loss = np.mean((hx - full_trajs[k])**2)
    # ax = axs[k]
    # ax.plot(tt, hx[:,1], color='blue', label='x')
    # ax.plot(tt, full_trajs[k][:,1], '--', color='blue')
    # ax.plot(tt, hx[:,2], color='red', label='y')
    # ax.plot(tt, full_trajs[k][:,2], '--', color='red')
    # ax.set_title(sites[k])
    # ax.legend()
# fig.tight_layout()
# plt.show()


