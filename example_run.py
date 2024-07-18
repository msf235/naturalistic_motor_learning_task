import humanoid2d
import mujoco as mj
import numpy as np
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
states = np.load('./data/phase_1/ball_grab/ball_grab_2_states.npy')
data.qpos[:] = states[0, :model.nq]
data.qvel[:] = states[0, model.nq:]

for k, ctrl in enumerate(ctrls):
    mj.mj_step1(model, data)
    data.ctrl[:] = ctrl
    mj.mj_step2(model, data)
    env.render()

time.sleep(3)
env.reset()
for k, state in enumerate(states):
    data.qpos[:] = state[:model.nq]
    data.qvel[:] = state[model.nq:]
    mj.mj_forward(model, data)
    env.render()
