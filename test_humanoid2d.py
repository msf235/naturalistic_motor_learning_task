import humanoid2d as h2d
import baseball_lqr as lqr
import numpy as np
import sim_utils as util
import mujoco as mj
import sys

seed = 2
rng = np.random.default_rng(seed)

Tk = 50

# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(render_mode='human', frame_skip=1)
env.reset(seed=seed)
model = env.model
data = env.data
# with open('humanoid2d.xml', 'r') as f:
    # xml = f.read()
# model = mj.MjModel.from_xml_path('humanoid.xml')
# data = mj.MjData(model)
# data1.qpos[0] = 2.5
# data2 = mj.MjData(model)

# Get noise
CTRL_STD = .05       # actuator units
# CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds
width = int(CTRL_RATE/model.opt.timestep)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
noise = util.FilteredNoise(model.nu, kernel, rng, CTRL_STD)
noisev = noise.sample(Tk-1)

### Get initial stabilizing controls
util.reset(model, data, 10)
qs, ctrls, K = lqr.get_stabilized_ctrls(model, data, Tk, noisev)
util.reset(model, data, 10)
for k in range(Tk-1):
    env.step(ctrls[k] + noisev[k])

util.reset(model, data, 10)
qs2 = util.forward_sim(model, data, ctrls + noisev)
print(np.allclose(qs, qs2))
breakpoint()

# Gradient descent

joints = lqr.get_joint_names(model)
right_arm_j = joints['right_arm_joint_inds']
right_arm_a = joints['right_arm_act_inds']
other_a = joints['non_right_arm_act_inds']

while True:
    sites1 = data.site('hand_right').xpos
    sites2 = data.site('target').xpos
    dlds = sites1 - sites2
    C = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)

    dldq = C.T @ dlds
    lams_fin = dldq

    reset(model, data, 10)
    grads = util.traj_deriv(model, data, qs, qvels, ctrls,
                            lams_fin, losses, fixed_act_inds=other_a)
    ctrls[:,right_arm_a] = ctrls[:, right_arm_a] - 20*grads[:Tk-1]
    print(ctrls[:5, right_arm_a])
    print(ctrls[-5:, right_arm_a])
    # breakpoint()

    env.reset(seed=seed)
    reset(model, data, 10)
    joints = lqr.get_joint_names(model)
    right_arm_j = joints['right_arm_joint_inds']
    qpos0n = qpos0.copy()
    for k in range(Tk-1):
        if k % 10 == 0:
            qpos = data.qpos.copy()
            # data.qpos[:] = qpos0
            qpos0n[right_arm_j] = data.qpos[right_arm_j]
            data.qpos[:] = qpos0n
            ctrl0 = lqr.get_ctrl0(model, data)
            data.ctrl[:] = ctrl0
            K = lqr.get_feedback_ctrl_matrix(model, data)
            data.qpos[:] = qpos
        ctrl = lqr.get_lqr_ctrl_from_K(model, data, K, qpos0n, ctrl0)
        ctrl[right_arm_a] = ctrls[k, right_arm_a]
        # ctrl[5] = -2
        # ctrl[6] = -2
        out = env.step(ctrl + CTRL_STD*noise.sample())
        observation, reward, terminated, __, info = out
        qs[k+1] = observation[:model.nq]
        qvels[k+1] = observation[model.nq:]
    # breakpoint()
