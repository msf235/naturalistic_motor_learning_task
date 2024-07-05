import humanoid2d
import opt_utils
import numpy as np
import sim_util as util
import sys
from pathlib import Path
import pickle as pkl
import arm_targ_traj as arm_t
from matplotlib import pyplot as plt
import torch
import mujoco as mj
import time
import config

args = config.get_arg_parser().parse_args()
vargs = vars(args)
print(args)
params = config.get_config(args.configfile)['params']
print(params)

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

out_f = Path(args.savefile)
out_f.parent.mkdir(parents=True, exist_ok=True)

Tf = params['Tf']

CTRL_STD = 0
CTRL_RATE = 1

rerun1 = False
# rerun1 = True

if args.render:
    render_mode = 'human'
else:
    # render_mode = 'rgb_array'
    render_mode = 'None'

keyframe = 'wide'

env = humanoid2d.Humanoid2dEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=params['reset_noise_scale'],
    xml_file=params['xml_file'],
    keyframe_name='wide',)
model = env.model
data = env.data

dt = model.opt.timestep
burn_step = int(.1 / dt)
reset = lambda : opt_utils.reset_with_lqr(env, args.seed, burn_step, 2*burn_step)

ctrls_burn_in = reset()

Tk = int(Tf / dt)

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

bodyj = joints['body']['body_dofs']

out_idx = arm_t.get_idx_sets(env, params['name'])
out_time = arm_t.get_times(env, params['name'], Tf)

out_traj = arm_t.make_traj_sets(env, params['name'], Tk)
sites = out_traj['sites']
targ_trajs = out_traj['target_trajs']

noisev = arm_t.make_noisev(model, args.seed, Tk, CTRL_STD, CTRL_RATE)

n = len(sites)
nr = range(n)

tt = np.arange(0, Tf, dt)

t_incr = params['t_incr']
amnt_to_incr = int(t_incr/dt)
grad_update_every = params['grad_update_every']
grad_trunc_tk = int(params['grad_window_t']/(grad_update_every*dt))

if rerun1 or not out_f.exists():
    ### Get initial stabilizing controls
    reset()
    stab_ctrls_idx = {k: out_idx[k] for k in
                      ['let_go_ids', 'contact_check_list',
                       'adh_ids']}
    stab_ctrls_idx.update({'let_go_times': out_time['let_go_times']})
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), acts['not_adh'],
        bodyj,
        free_ctrls=np.zeros((Tk, len(acts['adh']))),
        **stab_ctrls_idx,
    )[:2]
    # ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
    reset()
    # arm_t.forward_to_contact(env, ctrls, render=True)
    # reset()
    # breakpoint()

    arm_targ_params = {k: params[k] for k in
                       ['max_its', 'optimizer', 'lr', 'lr2', 'it_lr2',
                        'grad_update_every', 'incr_every',
                        # 't_incr',
                        'grab_phase_it',
                        # 'grad_window_t'
                       ]}
    ctrls, lowest_losses = arm_t.arm_target_traj(
        env,
        **out_traj,
        **arm_targ_params,
        **out_idx,
        **out_time,
        ctrls=ctrls, grad_trunc_tk=grad_trunc_tk,
        seed=args.seed, ctrl_rate=CTRL_RATE, ctrl_std=CTRL_STD,
        Tk=Tk,
        keep_top=10,
        amnt_to_incr=amnt_to_incr,
        phase_2_it=params['max_its']+1,
    )
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'lowest_losses': lowest_losses,
                  'ctrls_burn_in': ctrls_burn_in}, f)
else:
    with open(out_f, 'rb') as f:
        load_data = pkl.load(f)
    ctrls = load_data['ctrls']
    ctrls_burn_in = load_data['ctrls_burn_in']
    lowest_losses = load_data['lowest_losses']

ctrls = lowest_losses.peekitem(0)[1][1]
Te = 3
Tke = int(3 / dt)
ctrls_end = np.zeros((Tke, model.nu))
ctrls_full = np.vstack((ctrls, ctrls_end))
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
    arm_t.forward_with_sites(env, ctrls_full, sites, render=True)
    time.sleep(10)
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


