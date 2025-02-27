import joblib
import pickle as pkl
import numpy as np
import basic_env
import config
import arm_targ_traj as arm_t
import opt_utils

args = config.get_arg_parser().parse_args()
vargs = vars(args)
# config_name = args.configfile.split("/")[-1].split(".")[0]
params = config.get_config(args.configfile)["params"]
config_name = params["name"]
# Since numbers in scientific notation are converted to a string from yaml,
# need to convert these to a number.
params = {k: config.inp_to_num(v) for k, v in params.items()}

data_file = "output/data_latest.pkl"
with open(data_file, "rb") as f:
    data = pkl.load(f)

keyframe = "tpose1"
render_mode = "None"

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}

env = basic_env.BasicEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    xml_file=params["xml_file"],
    keyframe_name=keyframe,
)
model = env.model
data = env.data
ctrls = data["ctrl"]


def ret_fn(model, data):
    ret_dict = {}
    ret_dict.update(
        {
            # "site_dict": site_dict,
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(),
            "quat": data.quat.copy(),
        }
    )
    return ret_dict


arm_t.forward_and_collect_data(env, ctrls, ret_fn=None, render=False)


data_new = {}
data_new["0"] = {}
data_new["0"]["qs"] = data["qpos"]
data_new["0"]["vs"] = data["qvel"]
data_new["0"]["ctrls"] = data["ctrl"]
data_new["0"]["sensors"] = []
data_new["0"]["xquats"] = []
