# import humanoid2d
import basic_env
import opt_utils
import numpy as np
import sim_util as util
import shutil
from pathlib import Path
import pickle as pkl
import arm_targ_traj as arm_t
from matplotlib import pyplot as plt
import config

args = config.get_arg_parser().parse_args()
vargs = vars(args)
config_name = args.configfile.split("/")[-1].split(".")[0]
params = config.get_config(args.configfile)["params"]
# Since numbers in scientific notation are converted to a string from yaml,
# need to convert these to a number.
params = {k: config.inp_to_num(v) for k, v in params.items()}

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

name = args.name
out_f = (Path("output") / name).with_suffix(".pkl")

out_f.parent.mkdir(parents=True, exist_ok=True)

Tf = params["Tf"]
# Tf = 10*Tf

CTRL_STD = 0
CTRL_RATE = 1

if args.render:
    render_mode = "human"
else:
    # render_mode = 'rgb_array'
    render_mode = "None"

keyframe = "tpose1"

# env = humanoid2d.Humanoid2dEnv(
env = basic_env.BasicEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=params["reset_noise_scale"],
    xml_file=params["xml_file"],
    keyframe_name=keyframe,
)
model = env.model
data = env.data

dt = model.opt.timestep
# burn_step = int(.09 / dt)
# burn_step = int(.001 / dt)
burn_step = int(0.01 / dt)

# env.render()
# env.mujoco_renderer.viewer.add_marker(size=np.array([2, 2, 2]))
# env.render()


def reset():
    return opt_utils.reset_with_lqr(
        env,
        args.seed,
        burn_step,
        8 * burn_step,
        # 5000,
        params["balance_cost"],
        params["joint_cost"],
        params["root_cost"],
        params["foot_cost"],
        params["ctrl_cost"],
    )


ctrls_burn_in = reset()
# env.reset(seed=args.seed, options={"n_steps": 0, "render": False})
#
#
# def marker_fn(env, k):
#     env.mujoco_renderer.viewer.add_marker(
#         size=np.array([2, 2, 2]) / (k + 100) ** 0.5,
#         pos=np.array([-1, 0, 0]),
#         rgba=(1, 1, 0, 1),
#         type=mj.mjtGeom.mjGEOM_SPHERE,
#     )

# env.render()
# util.forward_sim_render(env, ctrls_burn_in, marker_fn)

Tk = int(Tf / dt)
tt = np.arange(0, Tf, dt)

joints = opt_utils.get_joint_ids(model)
acts = opt_utils.get_act_ids(model)

# body_dof = joints['body']['dofadrs_without_root']
body_dof = joints["body"]["dofadrs"]

out_idx = arm_t.get_idx_sets(env, params["name"])
sites = out_idx["sites"]
site_grad_idxs = out_idx["site_grad_idxs"]
stabilize_jnt_idx = out_idx["stabilize_jnt_idx"]
stabilize_act_idx = out_idx["stabilize_act_idx"]
out_time = arm_t.get_times(env, params["name"], Tf)


t_incr = params["t_incr"]
amnt_to_incr = int(t_incr / dt)
incr_every: int = params["incr_every"]
# incr_times = np.arange(amnt_to_incr, Tk, amnt_to_incr)
# incr_tk_left_intervals = np.arange(0, Tk, amnt_to_incr)
# incr_tk_end_intervals = np.arange(amnt_to_incr, Tk + 1, amnt_to_incr)

# targ_traj_masks = masks.make_basic_xpos_masks(incr_tk_end_intervals)


noisev = arm_t.make_noisev(model, args.seed, Tk, CTRL_STD, CTRL_RATE)

grad_update_every = params["grad_update_every"]
grad_trunc_tk = int(params["grad_window_t"] / (grad_update_every * dt))

Tke = int(params["t_after"] / dt)


if args.rerun or not out_f.exists():
    ### Get initial stabilizing controls
    reset()
    # stab_ctrls_idx = {k: out_idx[k] for k in
    # ['let_go_ids', 'contact_check_list',
    # 'adh_ids']}
    # stab_ctrls_idx.update({'let_go_times': out_time['let_go_times']})
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model,
        data,
        Tk,
        noisev,
        data.qpos.copy(),
        acts["not_adh"],
        body_dof,
        free_ctrls=np.zeros((Tk, len(acts["adh"]))),
        balance_cost=params["balance_cost"],
        joint_cost=params["joint_cost"],
        root_cost=params["root_cost"],
        foot_cost=params["foot_cost"],
        ctrl_cost=params["ctrl_cost"],
    )[:2]
    # ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
    # while True:
    reset()
    # util.forward_sim_render(env, ctrls)
    # arm_t.forward_to_contact(env, ctrls, render=True)
    # reset()

    ctrls, lowest_losses = arm_t.arm_target_traj(
        config_name=config_name,
        env=env,
        site_names=sites,
        site_grad_idxs=site_grad_idxs,
        stabilize_jnt_idx=stabilize_jnt_idx,
        stabilize_act_idx=stabilize_act_idx,
        # targ_trajs=targ_trajs,
        # targ_traj_masks=targ_traj_masks,
        # q_pos_masks=q_pos_masks,
        # q_vel_masks=q_vel_masks,
        # **traj_and_masks,
        # **arm_targ_params,
        ctrls=ctrls,
        grad_trunc_tk=grad_trunc_tk,
        seed=args.seed,
        ctrl_rate=CTRL_RATE,
        ctrl_std=CTRL_STD,
        Tk=Tk,
        max_its=params["max_its"],
        lr=params["lr"],
        lr2=params["lr2"],
        it_lr2=params["it_lr2"],
        keep_top=10,
        incr_every=incr_every,
        amnt_to_incr=amnt_to_incr,
        grad_update_every=params["grad_update_every"],
        phase_2_it=params["max_its"] + 1,
        plot_every=args.plot_every,
        render_every=args.render_every,
        optimizer=params["optimizer"],
        contact_check_list=out_idx["contact_check_list"],
        adh_ids=out_idx["adh_ids"],
        balance_cost=params["balance_cost"],
        joint_cost=params["joint_cost"],
        root_cost=params["root_cost"],
        # root_cost=0,  # For comparision. TODO: use line above instead
        foot_cost=params["foot_cost"],
        # foot_cost=1000,
        ctrl_cost=params["ctrl_cost"],
        let_go_times=out_time["let_go_times"],
        let_go_ids=out_idx["let_go_ids"],
        n_steps_adh=10,
        ctrl_reg_weight=params["ctrl_reg_weight"],
        joint_penalty_factor=params["joint_penalty_factor"],
        mask_decay_factor=params["mask_decay_factor"],
    )
    ctrls = np.vstack((ctrls, ctrls_burn_in))
    ctrls_end = np.zeros((Tke, model.nu))
    with open(out_f, "wb") as f:
        pkl.dump(
            {
                "ctrls": ctrls,
                "lowest_losses": lowest_losses,
                "ctrls_burn_in": ctrls_burn_in,
                "ctrls_end": ctrls_end,
            },
            f,
        )
else:
    with open(out_f, "rb") as f:
        load_data = pkl.load(f)
    ctrls = load_data["ctrls"]
    ctrls_burn_in = load_data["ctrls_burn_in"]
    lowest_losses = load_data["lowest_losses"]


ctrls = lowest_losses.peekitem(0)[1][1]
ctrls_end = np.zeros((Tke, model.nu))
reset()
hxs, qs = arm_t.forward_with_sites(env, ctrls, sites, render=False)
fig, axs = plt.subplots(2, len(sites), figsize=(8, 4))
if len(sites) == 1:
    axs = axs.reshape((2, 1))
q_targs_masked = []
qs_list = []
for k in range(len(sites)):
    q_targs_masked_tmp = traj_and_masks["q_targs"][k].copy()
    q_targs_masked_tmp[traj_and_masks["q_targ_masks"][k] == 0] = np.nan
    q_targs_masked.append(q_targs_masked_tmp)
    qs_tmp = qs.copy()
    qs_tmp[traj_and_masks["q_targ_masks"][k] == 0] = np.nan
    qs_list.append(qs_tmp)
arm_t.show_plot(
    axs,
    hxs,
    tt,
    traj_and_masks["targ_trajs"],
    traj_and_masks["targ_traj_masks"],
    sites,
    out_idx["site_grad_idxs"],
    qvals=qs_list,
    qtargs=q_targs_masked,
)
plt.show()
show_sim_cnt = 0
# while True:
# print("shown simulation {} times".format(show_sim_cnt))
# reset()
# arm_t.forward_with_sites(env, ctrls_full, sites, render=True)
# time.sleep(2)
# show_sim_cnt += 1


phase = args.task_phase
datadir = Path("data") / name
datadir.mkdir(parents=True, exist_ok=True)

shutil.copy("humanoid.xml", datadir)
shutil.copy("humanoid_and_basic.xml", datadir)
shutil.copy("basic_scene.xml", datadir)

ctrls_full = np.vstack((ctrls, ctrls_end))
reset()
qs, vs, ss = util.forward_sim(model, data, ctrls_full)
states = np.hstack((qs, vs))

np.save(datadir / "states_{}.npy".format(args.seed), states)

if phase in (1, 5):
    np.save(datadir / "ctrls_{}.npy".format(args.seed), ctrls_full)
    np.save(datadir / "sensors_{}.npy".format(args.seed), ss)
