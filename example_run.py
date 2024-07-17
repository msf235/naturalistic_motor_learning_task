import humanoid2d
import opt_utils
import numpy as np
import sim_util as util
import sys
from pathlib import Path
import pickle as pkl
import arm_targ_traj as arm_t
import basic_movements as bm
from matplotlib import pyplot as plt
import torch
import mujoco as mj
import time

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -10.0,
    "azimuth": 180,
}


### Set things up

render_mode = 'human'
# render_mode = 'rgb_array'

keyframe = 'wide'

# Create a Humanoid2dEnv object
env = humanoid2d.Humanoid2dEnv(
    render_mode=render_mode,
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG,
    reset_noise_scale=0,
    xml_file='./data/phase_1/ball_grab/humanoid_and_baseball.xml',
    keyframe_name='wide',)
model = env.model
data = env.data

ctrls = np.load('./data/phase_1/ball_grab/ball_grab_2_ctrls.npy')

while True:
    for k, ctrl in enumerate(ctrls):
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrl
        mj.mj_step2(model, data)
        env.render()

    time.sleep(10)
