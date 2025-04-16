# This one just sets the position of the racket manually
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
from scipy.spatial.transform import Rotation as sRot

args = config.get_arg_parser().parse_args()
vargs = vars(args)


# config_name = args.configfile.split("/")[-1].split(".")[0]
# Since numbers in scientific notation are converted to a string from yaml,
# need to convert these to a number.
#
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


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


def make_qpos_dicts():
    init_qpos = {
        47: -0.022,
        48: 0.310,
        49: -0.3,
        51: -0.3,
        53: -0.2,
        54: 0.15,
        56: 0.04,
        57: 0.22,
        58: 0.3,
        59: 0.10,
        60: 0.2,
        62: -0.022,
        63: 0.310,
        64: -0.3,
        66: 0.3,
        68: 0.2,
        69: -0.1,
        71: -0.04,
        72: -0.22,
        73: 0.3,
        74: -0.2,
        75: -0.2,
    }

    # breakpoint()


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
racket_qpos = joints["racket"]["qpos_adrs"]["all"]
ball_qpos = joints["ball"]["qpos_adrs"]["all"]

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


sqrt2_over_2 = np.sqrt(2) / 2

q_Id = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation
q_x = np.array([sqrt2_over_2, sqrt2_over_2, 0.0, 0.0])  # 90° about x
q_y = np.array([sqrt2_over_2, 0.0, sqrt2_over_2, 0.0])  # 90° about y
q_z = np.array([sqrt2_over_2, 0.0, 0.0, sqrt2_over_2])  # 90° about z


def iterate_through_qposs(
    env,
    render=True,  # Can also be a callable (function) which will be called to render
    qpos_dicts=None,
):
    model = env.model
    data = env.data
    if callable(render):
        render_fn = render
        render = True
    else:
        render_fn = env.render
    for qpos_dict in qpos_dicts:
        for idx, val in qpos_dict.items():
            data.qpos[idx] = val
        mj.mj_forward(model, data)

        hand_xquat = data.body("R_Hand").xquat
        xquat = quaternion_multiply(hand_xquat, q_y)
        xquat = quaternion_multiply(xquat, q_x)
        racket_targ_xpos = data.site("racket_center_grab").xpos
        data.qpos[racket_qpos[:3]] = racket_targ_xpos
        data.qpos[racket_qpos[3:]] = xquat

        hand_xquat = data.body("L_Hand").xquat
        xquat = quaternion_multiply(hand_xquat, q_y)
        xquat = quaternion_multiply(xquat, q_x)
        ball_targ_xpos = data.site("ball_center_grab").xpos
        data.qpos[ball_qpos[:3]] = ball_targ_xpos
        data.qpos[ball_qpos[3:]] = xquat

        mj.mj_forward(model, data)
        render_fn()
        input("Press Enter to continue...")


# (
#     sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
#     * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
# )
# .as_quat()
# .reshape(N, -1, 4)

# stab_ctrls_idx = {k: out_idx[k] for k in
# ['let_go_ids', 'contact_check_list',
# 'adh_ids']}
# stab_ctrls_idx.update({'let_go_times': out_time['let_go_times']})


if __name__ == "__main__":
    # 1: .022, 2: -.310, 3: -.3, 5: -.3, 7: -.2, 8: .15, 10: .04, 11: .22, 12: .3, 13: .10, 14: .2, 16: -.022, 17: .310, 18: -.3, 20: .3, 22: .2, 23: -.1, 25: -.04, 26: -.22, 27: .3, 28: -.2, 29: -.2

    init_qpos = {
        47: 0.022,
        48: -0.310,
        49: -0.3,
        51: -0.3,
        53: -0.2,
        54: 0.15,
        56: 0.04,
        57: 0.22,
        58: 0.3,
        59: 0.10,
        60: 0.2,
        62: -0.022,
        63: 0.310,
        64: -0.3,
        66: 0.3,
        68: 0.2,
        69: -0.1,
        71: -0.04,
        72: -0.22,
        73: 0.3,
        74: -0.2,
        75: -0.2,
    }
    qposs = [init_qpos]
    joint_poss = np.round(0.310 - np.linspace(0, 1, 20), 4)
    for joint_pos in joint_poss:
        qposs.append({63: joint_pos})
    iterate_through_qposs(env, render_fn, qposs)
# ctrls[:, tennis_idxs['adh_left_hand']] = left_adh_act_vals
# while True:
