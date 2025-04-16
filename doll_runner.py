import sim_util as util
import basic_env
import opt_utils
import numpy as np
from pathlib import Path
import pickle as pkl
import arm_targ_traj as arm_t
from matplotlib import pyplot as plt
import config
import mujoco as mj
import mujoco.viewer

args = config.get_arg_parser().parse_args()
vargs = vars(args)
# config_name = args.configfile.split("/")[-1].split(".")[0]
# Since numbers in scientific notation are converted to a string from yaml,
# need to convert these to a number.


CTRL_STD = 0
CTRL_RATE = 1

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

# I'm assuming that if there is a second phase, lr_2 is always defined

render_mode = None
# render_mode = "human"
keyframe = "tpose1"

env = basic_env.BasicEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    xml_file="./model_files/humanoid_and_tennis.xml",
    keyframe_name=keyframe,
)
model = env.model
data = env.data

dt = model.opt.timestep
burn_step = int(0.01 / dt)


def make_ctrls():
    fid = open("ctrls", "w")
    init_str = "16: -.022, 17: .310, 18: -.3, 20: .3, 22: .2, 23: -.1, 25: -.04, 26: -.22, 27: .3, 28: -.2, 29: -.2 | 150"
    # add 46
    # init_str = "62: -.022, 63: .310, 64: -.3, 66: .3, 68: .2, 69: -.1, 71: -.04, 72: -.22, 73: .3, 74: -.2, 75: -.2 | 150"
    fid.write(init_str + "\n")
    joint_poss = np.round(0.310 - np.linspace(0, 0.1, 100), 4)
    for joint_pos in joint_poss:
        row_str = "17: " + str(joint_pos) + " | 50"
        fid.write(row_str + "\n")
    joint_poss2 = np.ones(1000) * joint_poss[-1]
    for joint_pos in joint_poss2:
        row_str = "17: " + str(joint_pos) + " | 50"
        fid.write(row_str + "\n")
    fid.close()
    # breakpoint()


def reset():
    return opt_utils.reset_with_lqr(
        env,
        args.seed,
        burn_step,
        8 * burn_step,
        # 5000,
        0,
        5e8,
        1e4,
        0,
        1,
    )


ctrls_burn_in = reset()
# env.reset(seed=args.seed, options={"n_steps": 0, "render": False})
# env.render()
# # m = int(ctrls_burn_in.shape[1] // 4)
# # ctrls_burn_in[:, :m] = 0
# util.forward_sim_render(env, ctrls_burn_in)

joints = opt_utils.get_joints(model)
acts = opt_utils.get_act_ids(model)

# body_qpos = joints['body']['qpos_adrs_without_root']
# body_qpos = joints["body"]["qpos_adrs"]
body_ids = joints["body"]["ids"]

out_idx = arm_t.get_idx_sets(env, "tennis_serve")

sites = out_idx["sites"]
site_grad_idxs = out_idx["site_grad_idxs"]
stabilize_jnt_idx = out_idx["stabilize_jnt_idx"]
stabilize_act_idx = out_idx["stabilize_act_idx"]
Tf = 10
Tk = int(Tf / dt)
out_time = arm_t.get_times(
    env, "tennis_serve", Tf
)  # TODO: check with varying Tk_left_3


# incr_times = np.arange(amnt_to_incr, Tk, amnt_to_incr)
# incr_tk_left_intervals = np.arange(0, Tk, amnt_to_incr)
# incr_tk_end_intervals = np.arange(amnt_to_incr, Tk + 1, amnt_to_incr)

# targ_traj_masks = masks.make_basic_xpos_masks(incr_tk_end_intervals)


noisev = arm_t.make_noisev(model, args.seed, Tk, CTRL_STD, CTRL_RATE)
# contact_check_list = [
#     ["racket_node_1", HAND_STR_RIGHT + "_node_1"],
#     ["racket_node_2", HAND_STR_RIGHT + "_node_2"],
#     ["ball_core", HAND_STR_LEFT + "_core"],
# ]

acts_arms = acts["left_arm"] + acts["right_arm"]
free_acts = acts_arms + acts["adh"]
stabilize_jnt_idx = out_idx["stabilize_jnt_idx"]
stabilize_act_idx = out_idx["stabilize_act_idx"]
not_stabilize_act_idx = [k for k in range(model.nu) if k not in stabilize_act_idx]

arm_dof = (
    joints["body"]["qpos_adrs"]["left_arm"] + joints["body"]["qpos_adrs"]["right_arm"]
)
arm_ids = joints["body"]["ids"]["left_arm"] + joints["body"]["ids"]["right_arm"]

print("Length of arm_dof: ", len(arm_dof))
arm_names = {k: model.joint(arm_ids[k]).name for k in range(len(arm_ids))}
print(arm_names)

# breakpoint()

viewer = mujoco.viewer.launch_passive(model, data)


def render_fn():
    # env.render()
    viewer.sync()


free_ctrls = np.zeros((Tk, len(not_stabilize_act_idx)))

with open("ctrls", "r") as jnt_file:
    jnt_data = jnt_file.readlines()

balance_cost = 0
joint_cost = 5e6
root_cost = 1e4
foot_cost = 0
ctrl_cost = 1
n_steps_adh = 100


def get_lqr(model, data):
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model,
        data,
        Tk,
        noisev,
        data.qpos.copy(),
        stabilize_act_idx,
        K_update_interv=10000,
        stable_jnt_ids=stabilize_jnt_idx,
        free_ctrls=free_ctrls,
        balance_cost=balance_cost,
        joint_cost=joint_cost,
        root_cost=root_cost,
        foot_cost=foot_cost,
        ctrl_cost=ctrl_cost,
        n_steps_adh=n_steps_adh,
    )[:2]
    return ctrls, K


def forward_with_dynamic_adhesion(
    env,
    ctrls,
    noisev=None,
    render=True,  # Can also be a callable (function) which will be called to render
    let_go_times=[],
    let_go_ids=[],
    n_steps_adh=40,
    contact_check_list=[],
    adh_ids=[],
    lqr_ctrl0=None,
    lqr_stable_jnt_ids=None,
    lqr_act_ids=None,
):
    """Simulate dynamics according to ctrls, while performing dynamic adhesion. This adhesion will automatically ramp up adhesion actuators when
    contacts as defined by contact_check_list are detected."""
    model = env.model
    data = env.data
    if callable(render):
        render_fn = render
        render = True
    else:
        render_fn = env.render
    Tk = ctrls.shape[0]
    adh_ctrl = opt_utils.AdhCtrl(
        let_go_times, let_go_ids, n_steps_adh, contact_check_list, adh_ids
    )
    if noisev is None:
        noisev = np.zeros((Tk, model.nu))
    prev_jnt = [0] * len(arm_dof)
    step_cnt = 0
    n_steps = 0
    file_line = 0
    qpos0n = data.qpos.copy()
    for k in range(Tk):
        if step_cnt >= n_steps and file_line < len(jnt_data):
            step_cnt = 0
            res, res2 = jnt_data[file_line].split("|")
            # res = input(f"arm qpos: {prev_jnt}:\n")
            # res2 = input(f"num simulation steps (1):  \n")
            # if res2 == "":
            #     res2 = "1"
            n_steps = int(res2)
            if res != "":
                res = res.split(",")
                for p in res:
                    psplit = p.split(":")
                    index = int(psplit[0])
                    value = float(psplit[1])
                    prev_jnt[index] = value
            file_line += 1
            input("Press enter to continue.")
        breakpoint()
        for idx in range(len(prev_jnt)):
            data.qpos[arm_dof[idx]] = prev_jnt[idx]
        mj.mj_forward(model, data)
        if step_cnt == 0:
            lqr_K = opt_utils.get_feedback_ctrl_matrix(
                model,
                data,
                ctrl0,
                stable_jnt_ids=stabilize_jnt_idx,
                active_ctrl_ids=stabilize_act_idx,
                balance_cost=balance_cost,
                joint_cost=joint_cost,
                root_cost=root_cost,
                foot_cost=foot_cost,
                ctrl_cost=ctrl_cost,
            )
            # ctrls[k], lqr_K = get_lqr(model, data)
        ctrls[k], _, _ = adh_ctrl.get_ctrl(model, data, ctrls[k])
        print(ctrls[k, adh_ids])
        ctrls[k, lqr_act_ids] = opt_utils.get_lqr_ctrl_from_K(
            model, data, lqr_K, qpos0n, lqr_ctrl0, lqr_stable_jnt_ids
        )
        util.step(model, data, ctrls[k] + noisev[k])
        if render:
            render_fn()
        step_cnt += 1
    return ctrls

    # stab_ctrls_idx = {k: out_idx[k] for k in
    # ['let_go_ids', 'contact_check_list',
    # 'adh_ids']}
    # stab_ctrls_idx.update({'let_go_times': out_time['let_go_times']})


if __name__ == "__main__":
    make_ctrls()
    reset()

    ctrls, K = get_lqr(model, data)

    reset()
    ctrl0 = opt_utils.get_ctrl0(model, data, stabilize_jnt_idx, stabilize_act_idx)
    # ctrls[:, stabilize_act_idx] = ctrl0
    reset()
    ctrls = forward_with_dynamic_adhesion(
        env,
        ctrls,
        noisev,
        render=render_fn,
        n_steps_adh=n_steps_adh,
        contact_check_list=out_idx["contact_check_list"],
        adh_ids=out_idx["adh_ids"],
        lqr_ctrl0=ctrl0,
        lqr_stable_jnt_ids=stabilize_jnt_idx,
        lqr_act_ids=stabilize_act_idx,
    )
# ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
# while True:
