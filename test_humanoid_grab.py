import basic_env
import opt_utils as opt_utils
import numpy as np
import arm_targ_traj
import sim_util as util
import mujoco as mj
import sys
import pickle as pkl

### Set things up
seed = 2
rng = np.random.default_rng(seed)

Tk = 50

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

# Create a Humanoid2dEnv object
let_go_times = [2610]
let_go_ids = [70]
n_steps_adh = 200
contact_check_list = [["ball_core", "L_Hand_core"], ["racket_core", "R_Hand_core"]]
env = basic_env.BasicEnv(
    render_mode="human",
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0.0,
    xml_file="./model_files/humanoid_and_tennis.xml",
    keyframe_name="tpose1",
)
env.reset(seed=seed)
model = env.model
data = env.data

with open("output/data_latest.pkl", "rb") as f:
    saved_data = pkl.load(f)

qposs = saved_data["qpos"]
targ = saved_data["trajectory_target"]
site_names = saved_data["site_names"]
ctrls = saved_data["ctrl"]
adh_ids = [69, 70]

# let_go_times,
# let_go_ids,
# n_steps_adh,
# contact_check_list,
# adh_ids,
arm_targ_traj.forward_with_dynamic_adhesion(
    env,
    ctrls,
    let_go_times=let_go_times,
    let_go_ids=let_go_ids,
    n_steps_adh=n_steps_adh,
    contact_check_list=contact_check_list,
    adh_ids=adh_ids,
)
breakpoint()

joints = opt_utils.get_joint_names(model)
right_arm_j = joints["right_arm_joint_inds"]
right_arm_a = joints["right_arm_act_inds"]
other_a = joints["non_right_arm_act_inds"]
adh = joints["adh_right_hand"]


def show_forward_sim(model, data, ctrls):
    for k in range(ctrls.shape[0] - 1):
        env.step(ctrls[k])
