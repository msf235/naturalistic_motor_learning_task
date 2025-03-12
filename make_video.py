import sys
import pickle as pkl
import util
import sim_util
import basic_env
import arm_targ_traj as arm_t
from pathlib import Path
import mujoco as mj

args = sys.argv[1:]

with open(args[0], "rb") as f:
    data_load = pkl.load(f)

if args[4] == "human":
    print("Human")
    render_mode = "human"
else:
    render_mode = "None"

# xml_path = Path("..") / Path(data_load["model_file_location"])
env = basic_env.BasicEnv(
    render_mode=render_mode,
    frame_skip=1,
    reset_noise_scale=data_load["reset_noise_scale"],
    # reset_noise_scale=0,
    xml_file=data_load["model_file_location"],
    # xml_file="./model_files/humanoid_and_tennis.xml",
    keyframe_name=data_load["keyframe"],
    # keyframe_name="tpose1",
)
model = env.model
data = env.data
# sim_util.reset_state(model, )
data.qpos[:] = data_load["state0"]["qpos"].copy()
data.qvel[:] = data_load["state0"]["qvel"].copy()
# data_to.qacc[:] = data_from.qacc.copy()
# data.act[:] = data_from.act.copy()
# data.ctrl[:] = data_from.ctrl.copy()
data.time = data_load["state0"]["time"]
mj.mj_forward(model, data)
# ctrls = data_load["ctrls_trunc"]
ctrls = data_load["best_pair"][1]
breakpoint()
targ = data_load["trajectory_target"]
site_names = data_load["site_names"]

if args[4] == "human":
    while True:
        env.render()
        render_class = util.targetRender(env, targ, site_names)
        render_fn = render_class.render
        arm_t.forward_and_collect_data(env, ctrls, ret_fn=None, render=render_fn)
        data.qpos[:] = data_load["state0"]["qpos"].copy()
        data.qvel[:] = data_load["state0"]["qvel"].copy()
        data.time = data_load["state0"]["time"]
        mj.mj_forward(model, data)
else:

    def ret_fn(model, data):
        return {"qpos": data.qpos.copy()}

    sim_data = arm_t.forward_and_collect_data(env, ctrls, ret_fn=ret_fn, render=False)

    qposs = sim_data["qpos"]
    util.make_video_of_motion(args[1], qposs, args[2], targ, site_names, float(args[3]))
