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
from pathlib import Path

args = config.get_arg_parser().parse_args()
vargs = vars(args)


out_dir = Path("out_qpos/")
out_dir.mkdir(parents=True, exist_ok=True)

render = False


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
# arm_names = {k: model.joint(arm_ids[k]).name for k in range(len(arm_ids))}
arm_names = {arm_dof[k]: model.joint(arm_id).name for k, arm_id in enumerate(arm_ids)}
arm_dofs = {model.joint(arm_id).name: arm_dof[k] for k, arm_id in enumerate(arm_ids)}
print(arm_names)

if render:
    viewer = mujoco.viewer.launch_passive(model, data)


if render:

    def render_fn():
        # env.render()
        viewer.sync()
        input("Press Enter to continue...")
else:

    def render_fn():
        pass


sqrt2_over_2 = np.sqrt(2) / 2

q_Id = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation
q_x = np.array([sqrt2_over_2, -sqrt2_over_2, 0.0, 0.0])  # -90° about x
q_y = np.array([sqrt2_over_2, 0.0, -sqrt2_over_2, 0.0])  # -90° about y
q_z = np.array([sqrt2_over_2, 0.0, 0.0, sqrt2_over_2])  # 90° about z

qpos_data = []


def iterate_through_qposs(
    env,
    render=True,  # Can also be a callable (function) which will be called to render
    qpos_dicts=None,
    follow=False,
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
            data.qpos[arm_dofs[idx]] = val
        mj.mj_forward(model, data)

        if follow:
            xquat = data.body("R_Hand").xquat
            xquat = quaternion_multiply(xquat, q_y)
            xquat = quaternion_multiply(xquat, q_x)
            # xquat = quaternion_multiply(hand_xquat, q_z)
            racket_targ_xpos = data.site("racket_center_grab").xpos
            data.qpos[racket_qpos[:3]] = racket_targ_xpos
            data.qpos[racket_qpos[3:]] = xquat

            hand_xquat = data.body("L_Hand").xquat
            # xquat = quaternion_multiply(hand_xquat, q_y)
            xquat = quaternion_multiply(xquat, q_x)
            xquat = quaternion_multiply(hand_xquat, q_y)
            ball_targ_xpos = data.site("ball_center_grab").xpos
            data.qpos[ball_qpos[:3]] = ball_targ_xpos
            data.qpos[ball_qpos[3:]] = xquat

            mj.mj_forward(model, data)
        render_fn()
        qpos_data.append(data.qpos.copy())
        # input("Press Enter to continue...")


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
#
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


def interps(v1, v2, ss):
    ret_val = []
    for s in ss:
        new_dict = {k: (1 - s).item() * v1[k] + s.item() * v2[k] for k in v1.keys()}
        ret_val.append(new_dict)
    return ret_val


def zero_non_specified(d1):
    for key in arm_dofs.keys():
        if key not in d1.keys():
            d1[key] = 0


if __name__ == "__main__":
    # 1: .022, 2: -.310, 3: -.3, 5: -.3, 7: -.2, 8: .15, 10: .04, 11: .22, 12: .3, 13: .10, 14: .2, 16: -.022, 17: .310, 18: -.3, 20: .3, 22: .2, 23: -.1, 25: -.04, 26: -.22, 27: .3, 28: -.2, 29: -.2
    T = 3
    framerate = 30
    frames = T * framerate
    frames_per_phase = int(frames / 5)
    final_phase_frames = frames - frames_per_phase * 4
    ctrls_burn_in = reset()

    init_qpos1 = {k: 0.0 for k in arm_dofs.keys()}  # Initial zero position
    init_qpos2 = {  # Grabbing position
        "L_Thorax_y": 0.022,
        "L_Thorax_z": -0.310,
        "L_Shoulder_x": -0.3,
        "L_Shoulder_z": -0.3,
        "L_Elbow_y": -0.2,
        "L_Elbow_z": 0.15,
        "L_Wrist_y": 0.04,
        "L_Wrist_z": 0.22,
        "L_Hand_x": 0.3,
        "L_Hand_y": 0.10,
        "L_Hand_z": 0.2,
        "R_Thorax_y": -0.022,
        "R_Thorax_z": 0.310,
        "R_Shoulder_x": -0.3,
        "R_Shoulder_z": 0.3,
        "R_Elbow_y": 0.2,
        "R_Elbow_z": -0.1,
        "R_Wrist_y": -0.04,
        "R_Wrist_z": -0.22,
        "R_Hand_x": 0.3,
        "R_Hand_y": -0.2,
        "R_Hand_z": -0.2,
    }
    # qposs = [init_qpos1]
    zero_non_specified(init_qpos2)
    ss = np.linspace(0, 1, frames_per_phase)
    qposs = interps(init_qpos1, init_qpos2, ss)
    iterate_through_qposs(env, render_fn, qposs, follow=False)

    init_qpos3 = {  # Prep setup position
        "R_Wrist_x": -1.55,
        "R_Hand_x": -1.55,
        "R_Elbow_z": -0.75,
        "L_Wrist_x": -1.55,
        "L_Hand_x": -1.55,
        "L_Elbow_z": 0.15,
    }
    zero_non_specified(init_qpos3)
    ss = np.linspace(0, 1, frames_per_phase - 4)
    qposs = interps(init_qpos2, init_qpos3, ss)
    qposs += [init_qpos3] * 4  # Linger on the final prep position
    iterate_through_qposs(env, render_fn, qposs, follow=True)

    # init_qpos4 = {51: -0.1, 53: -1, 54: 1, 58: -2.0, 60: 0.4}
    init_qpos4 = {  # Throwing ball position
        "R_Wrist_x": -1.55,
        "R_Hand_x": -1.55,
        "R_Elbow_z": -0.75,
        "L_Shoulder_z": 0.5,
        "L_Elbow_z": 0.88,
        "L_Wrist_x": -1.55,
        "L_Hand_z": 0.42,
        "L_Hand_x": -1.55,
    }
    zero_non_specified(init_qpos4)
    ss = np.linspace(0, 1, frames_per_phase)
    qposs = interps(init_qpos3, init_qpos4, ss)
    iterate_through_qposs(env, render_fn, qposs, follow=True)

    init_qpos5 = {  # Hitting racket position
        "R_Wrist_x": -1.55,
        "R_Hand_x": -1.55,
        "R_Thorax_z": -0.6,
        "R_Shoulder_z": -0.82,
        "R_Elbow_z": -1,
        "R_Hand_z": -0.36,
    }
    zero_non_specified(init_qpos5)
    ss = np.linspace(0, 1, final_phase_frames)
    qposs = interps(init_qpos4, init_qpos5, ss)
    iterate_through_qposs(env, render_fn, qposs, follow=True)

    qpos_data = np.array(qpos_data)
    np.save(out_dir / "qpos_data.npy", qpos_data)
