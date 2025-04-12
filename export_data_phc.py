import pickle as pkl
import joblib
import numpy as np
from pathlib import Path
import basic_env
import config
import arm_targ_traj as arm_t
from scipy.spatial.transform import Rotation as sRot
import torch

# from poselib.poselib.skeleton.skeleton3d import (
#     SkeletonTree,
#     SkeletonMotion,
#     SkeletonState,
# )
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
# from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


# xquat_mujoco: shape [N, 4] or [N, 24, 4] in MuJoCo format [w, x, y, z]
# Convert to SciPy format [x, y, z, w]
def convert_mujoco_quat_to_scipy(xquat_mujoco):
    return np.concatenate([xquat_mujoco[..., 1:], xquat_mujoco[..., :1]], axis=-1)


# SMPL_BONE_ORDER_NAMES = [
#     "Pelvis",
#     "L_Hip",
#     "R_Hip",
#     "Torso",
#     "L_Knee",
#     "R_Knee",
#     "Spine",
#     "L_Ankle",
#     "R_Ankle",
#     "Chest",
#     "L_Toe",
#     "R_Toe",
#     "Neck",
#     "L_Thorax",
#     "R_Thorax",
#     "Head",
#     "L_Shoulder",
#     "R_Shoulder",
#     "L_Elbow",
#     "R_Elbow",
#     "L_Wrist",
#     "R_Wrist",
#     "L_Hand",
#     "R_Hand",
# ]
#
# SMPL_MUJOCO_NAMES = [
#     "Pelvis",
#     "L_Hip",
#     "L_Knee",
#     "L_Ankle",
#     "L_Toe",
#     "R_Hip",
#     "R_Knee",
#     "R_Ankle",
#     "R_Toe",
#     "Torso",
#     "Spine",
#     "Chest",
#     "Neck",
#     "Head",
#     "L_Thorax",
#     "L_Shoulder",
#     "L_Elbow",
#     "L_Wrist",
#     "L_Hand",
#     "R_Thorax",
#     "R_Shoulder",
#     "R_Elbow",
#     "R_Wrist",
#     "R_Hand",
# ]

# Load configuration and set parameters
args = config.get_arg_parser().parse_args()
vargs = vars(args)

upright_start = True

# robot_cfg = {
#     "mesh": False,
#     "rel_joint_lm": True,
#     "upright_start": upright_start,
#     "remove_toe": False,
#     "real_weight": True,
#     "real_weight_porpotion_capsules": True,
#     "real_weight_porpotion_boxes": True,
#     "replace_feet": True,
#     "masterfoot": False,
#     "big_ankle": True,
#     "freeze_hand": False,
#     "box_body": False,
#     "master_range": 50,
#     "body_params": {},
#     "joint_params": {},
#     "geom_params": {},
#     "actuator_params": {},
#     "model": "smpl",
# }
# smpl_local_robot = LocalRobot(
#
#     robot_cfg,
# )


def ret_fn(model, data):
    ret_data = {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "xpos": data.xpos.copy(),
        "xquats": data.xquat.copy(),
        "sensordata": data.sensordata[:].copy(),
    }
    return ret_data


def process_data(ctrls, env, t_after=0, ret_fn=ret_fn, label=""):
    Tke = int(t_after / model.opt.timestep)
    ctrls_full = np.vstack((ctrls, np.zeros((Tke, model.nu))))

    sim_data = arm_t.forward_and_collect_data(
        env, ctrls_full, ret_fn=ret_fn, render=False
    )
    sim_data["dt"] = env.model.opt.timestep
    sim_data["njnts"] = env.model.njnt
    return sim_data


def process_and_save_data(
    ctrls,
    env,
    datadir,
    t_after=0,
    label="",
):
    datadir.mkdir(parents=True, exist_ok=True)
    full_data = process_data(ctrls, env, t_after=t_after, label=label)
    joblib.dump(
        full_data,
        datadir / f"full_data{label}.pkl",
    )


def convert_mujoco_to_smpl(qpos):
    # qpos: [N, 76] = 3 (root trans) + 4 (root quat) + 23×3 axis-angle
    N = qpos.shape[0]

    smpl_2_mujoco = [
        SMPL_BONE_ORDER_NAMES.index(q)
        for q in SMPL_MUJOCO_NAMES
        if q in SMPL_BONE_ORDER_NAMES
    ]
    mujoco_2_smpl = [
        SMPL_MUJOCO_NAMES.index(q)
        for q in SMPL_BONE_ORDER_NAMES
        if q in SMPL_MUJOCO_NAMES
    ]

    # Leave off the global translation
    root_trans = qpos[:, :3]  # [N, 3]
    root_orient = sRot.from_quat(qpos[:, 3:7]).as_rotvec()  # [N, 3] <- [N,4]
    pose_aa_mj_flat = np.concatenate((root_orient, qpos[:, 7:]), axis=1)

    pose_aa_mj = pose_aa_mj_flat.reshape(-1, 24, 3)
    # pose_aa_smpl = pose_aa_mj[:, mujoco_2_smpl]
    pose_aa_smpl = pose_aa_mj[:, mujoco_2_smpl]
    pose_aa_smpl_flat = pose_aa_smpl.reshape(N, 72)

    pose_quat = (
        sRot.from_rotvec(pose_aa_smpl.reshape(-1, 3)).as_quat().reshape(N, 24, 4)
    )

    beta = np.zeros((16))
    gender_number, beta[:], gender = [0], 0, "neutral"
    # print("using neutral model")
    #
    # smpl_local_robot.load_from_skeleton(
    #     betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None
    # )
    # smpl_local_robot.write_xml(f"tmp/tmp_humanoid.xml")
    # skeleton_tree = SkeletonTree.from_mjcf(f"tmp/tmp_humanoid.xml")
    # root_trans_offset = (
    #     torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
    # )
    #
    # new_sk_state = SkeletonState.from_rotation_and_root_translation(
    #     skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
    #     torch.from_numpy(pose_quat),
    #     root_trans_offset,
    #     is_local=True,
    # )
    #
    # if robot_cfg["upright_start"]:
    #     pose_quat_global = (
    #         (
    #             sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
    #             * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    #         )
    #         .as_quat()
    #         .reshape(N, -1, 4)
    #     )  # should fix pose_quat as well here...
    #
    #     new_sk_state = SkeletonState.from_rotation_and_root_translation(
    #         skeleton_tree,
    #         torch.from_numpy(pose_quat_global),
    #         root_trans_offset,
    #         is_local=False,
    #     )
    #     pose_quat = new_sk_state.local_rotation.numpy()
    #
    #     pose_quat_global = new_sk_state.global_rotation.numpy()
    #     pose_quat = new_sk_state.local_rotation.numpy()

    joint_body_ids = [model.body(name).id for name in SMPL_BONE_ORDER_NAMES]
    pose_quat_global_wxyz = data["xquats"][::skip, joint_body_ids]  # shape: [24, 4]
    pose_quat_global_smpl = convert_mujoco_quat_to_scipy(pose_quat_global_wxyz)

    new_motion_out = {}
    new_motion_out["pose_quat_global"] = pose_quat_global_smpl
    new_motion_out["pose_aa"] = pose_aa_smpl_flat
    new_motion_out["fps"] = 30
    new_motion_out["pose_quat"] = pose_quat
    new_motion_out["trans_orig"] = root_trans
    new_motion_out["root_trans_offset"] = torch.from_numpy(0 * root_trans)
    new_motion_out["beta"] = beta
    new_motion_out["gender"] = "neutral"

    return new_motion_out


if __name__ == "__main__":
    # Example usage of the refactored functions
    # ctrls, ctrls_burn_in, env, model, data = run_experiment()
    name = args.name

    out_dir = Path(args.savedir) / name
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.load_it == -1:
        filename = "data_latest.pkl"
    else:
        filename = f"data_{args.load_it}.pkl"
    with open(out_dir / filename, "rb") as f:
        data_load = pkl.load(f)
    ctrls = data_load["best_pair"][1]

    if args.render:
        render_mode = "human"
    else:
        render_mode = "None"

    # Initialize environment
    env = basic_env.BasicEnv(
        render_mode=render_mode,
        frame_skip=1,
        reset_noise_scale=data_load["reset_noise_scale"],
        xml_file=data_load["model_file_location"],
        keyframe_name=data_load["keyframe"],
    )

    model = env.model
    data = env.data
    data = process_data(ctrls, env, label=f"_{name}_{args.load_it}")
    np.save("qpos.npy", data["qpos"])
    breakpoint()
    # framerate = int(round(1 / model.opt.timestep))
    # skip = int(framerate / 30)
    # # np.save("qpos.npy", qpos)
    # qpos = data["qpos"][::skip]
