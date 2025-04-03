import pickle as pkl
import joblib
import numpy as np
from pathlib import Path
import basic_env
import config
import opt_utils
import sim_util as util
import arm_targ_traj as arm_t
from matplotlib import pyplot as plt
import mujoco as mj

# Load configuration and set parameters
args = config.get_arg_parser().parse_args()
vargs = vars(args)


def ret_fn(model, data):
    ret_data = {
        "qs": data.qpos.copy(),
        "vs": data.qvel.copy(),
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
    full_data = {"0": sim_data}
    return full_data


def process_and_save_data(
    ctrls,
    env,
    datadir,
    t_after=0,
    label="",
):
    datadir.mkdir(parents=True, exist_ok=True)
    full_data = process_data(ctrls, env, t_after=t_after, label=label)
    joblib.dump(full_data, datadir / f"full_data{label}.pkl")


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
    breakpoint()

    model = env.model
    data = env.data

    process_and_save_data(
        ctrls, env, datadir=Path("data_phc") / name, label=f"_{name}_{args.load_it}"
    )
