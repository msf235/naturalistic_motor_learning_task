import humanoid2d as h2d
import baseball_lqr as lqr
import numpy as np
import sim_utils as util
import mujoco as mj

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

qpos0 = data.qpos.copy()
ctrl0 = lqr.get_ctrl0(model, data)
data.ctrl = ctrl0
K = lqr.get_feedback_ctrl_matrix(model, data)

# CTRL_STD = 0.05       # actuator units
CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds
width = int(CTRL_RATE/model.opt.timestep)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
noise = util.FilteredNoise(model.nu, kernel, 3*seed+7)

Tk = 20
qs = np.zeros((Tk, model.nq))
qs[0] = qpos0
vs = np.zeros((Tk, model.nq))
vs[0] = data.qvel.copy()
us = np.zeros((Tk, model.nu))
losses = np.zeros((Tk,))

for k in range(Tk-1):
    ctrl = lqr.get_lqr_ctrl_from_K(model, data, K, qpos0, ctrl0)
    out = env.step(ctrl + CTRL_STD*noise.sample())
    observation, reward, terminated, __, info = out
    qs[k+1] = observation[:model.nq]
    vs[k+1] = observation[model.nq:]

sites1 = data.site('hand_right').xpos
sites2 = data.site('target').xpos
dlds = sites1 - sites2
# curr_loss = .5*np.linalg.norm(dlds)**2
# Cs = np.zeros((Tk, 3, model.nv))
C = np.zeros((3, model.nv))
mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)

dldq = C.T @ dlds
# lams_fin = np.zeros((Tk, model.nu))
lams_fin = dldq

# lams[Tk-1,:nv] = dldq
# lams[Tk-1,2] += dldtheta

grads = util.traj_deriv(model, data, qs, vs, us, lams_fin, losses)

breakpoint()
