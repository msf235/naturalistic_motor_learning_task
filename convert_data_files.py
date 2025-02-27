import joblib
import pickle as pkl
import numpy as np
import basic_env
import config
import arm_targ_traj as arm_t
import opt_utils

data_file = "./output/tennis_new_output/data_latest.pkl"
with open(data_file, "rb") as f:
    data_load = pkl.load(f)

env = basic_env.BasicEnv(
    render_mode="None",
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
            "qs": data.qpos.copy(),
            "vs": data.qvel.copy(),
            "ctrls": data.ctrl.copy(),
            "xquats": data.xquat.copy(),
            "xpos": data.xpos.copy(),
            "sensors": data.sensordata.copy(),
        }
    )
    return ret_dict


sim_data = arm_t.forward_and_collect_data(env, ctrls, ret_fn=ret_fn, render=False)
sim_data["dt"] = model.opt.timestep
sim_data["njnts"] = model.njnt

out_data = {"0": sim_data}
joblib.dump(out_data, "./output/tennis_new_output/sim_data.pkl")
