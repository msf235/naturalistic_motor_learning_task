import humanoid2d as h2d
import baseball_lqr as lqr
import numpy as np
import sim_util as util

seed = 2

# # Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(render_mode='human')
env.reset(seed=seed)

# zac = 9*[0]
# for k in range(200):
    # env.step(zac)


# Test LQR controller
env.reset(seed=seed)
model = env.model
data = env.data


# Get initial stabilizing controls
## Burn in:
for k in range(10):
    env.step(np.zeros(model.nu))
    # mj.mj_step(model, data)

qpos0 = data.qpos.copy()
ctrl0 = lqr.get_ctrl0(model, data)
data.ctrl = ctrl0
K = lqr.get_feedback_ctrl_matrix(model, data)

CTRL_STD = 0.05       # actuator units
CTRL_RATE = 0.8       # seconds
width = int(CTRL_RATE/model.opt.timestep)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
noise = util.FilteredNoise(model.nu, kernel, 3*seed+7)

for k in range(2000):
    ctrl = lqr.get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0)
    env.step(ctrl + CTRL_STD*noise.sample())
    # mj.mj_step(model, data)
# breakpoint()
