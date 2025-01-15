import mujoco
import basic_env

# from mujoco import viewer
import numpy as np
from matplotlib import pyplot as plt
from mujoco_rendering import WindowViewer

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    # "distance": 4.0,
    "distance": 10,
    "lookat": np.array((0.0, 0.0, 1.15)),
    # "elevation": -20.0,
    "elevation": -10.0,
    "azimuth": 180,
}

keyframe = "tpose1"
env = basic_env.BasicEnv(
    render_mode="human",
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    xml_file="./model_files/humanoid_and_baseball.xml",
    keyframe_name=keyframe,
)
data = env.data
model = env.model

data0 = np.loadtxt("model_files/data_0.txt")
data1 = np.loadtxt("model_files/data_1.txt")
data2 = np.loadtxt("model_files/data_2.txt")
data3 = np.loadtxt("model_files/data_3.txt")
data.qpos[:] = data3

mujoco.mj_forward(model, data)
nonzero = np.nonzero(data.qpos)[0]
shifted = nonzero - 6
shifted = [s.item() for s in shifted if s > 0]
jnt_names = [model.joint(s).name for s in shifted]
jnt_vals = [data.qpos[s + 6].item() for s in shifted]

for k in range(1000):
    env.render()
