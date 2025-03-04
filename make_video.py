import sys
import pickle as pkl
import util
import basic_env
import arm_targ_traj as arm_t
from pathlib import Path

args = sys.argv[1:]

with open(args[0], "rb") as f:
    data_load = pkl.load(f)

# xml_path = Path("..") / Path(data_load["model_file_location"])
env = basic_env.BasicEnv(
    # render_mode="None",
    render_mode="human",
    frame_skip=1,
    reset_noise_scale=data_load["reset_noise_scale"],
    xml_file=data_load["model_file_location"],
    keyframe_name=data_load["keyframe"],
)
model = env.model
data = env.data
ctrls = data_load["ctrl"]


def ret_fn(model, data):
    ret_dict = {}
    ret_dict.update(
        {
            # "site_dict": site_dict,
            "qpos": data.qpos.copy(),
        }
    )
    return ret_dict


# ctrls_test = arm_t.forward_with_dynamic_adhesion(
#     env,
#     ctrls,
#     render=True,
#     n_steps_adh=100,
#     adh_ids=[69, 70],
#     contact_check_list=[["racket_core", "R_Hand_core"], ["ball_core", "L_Hand_core"]],
# )
# breakpoint()

sim_data = arm_t.forward_and_collect_data(env, ctrls, ret_fn=ret_fn, render=False)

qposs = sim_data["qpos"]
targ = data_load["trajectory_target"]
site_names = data_load["site_names"]
util.make_video_of_motion(args[1], qposs, args[2], targ, site_names, float(args[3]))
