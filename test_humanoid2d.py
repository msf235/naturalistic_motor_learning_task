import humanoid2d as h2d
import opt_utils as opt_utils
import numpy as np
import sim_utils as util
import mujoco as mj
import sys

### Set things up
seed = 2
rng = np.random.default_rng(seed)

Tk = 50

# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(
    render_mode='human',
    # render_mode='rgb_array',
    frame_skip=1)
env.reset(seed=seed)
model = env.model
data = env.data

joints = opt_utils.get_joint_names(model)
right_arm_j = joints['right_arm_joint_inds']
right_arm_a = joints['right_arm_act_inds']
other_a = joints['non_right_arm_act_inds']
sites1 = data.site('hand_right').xpos
sites2 = data.site('target').xpos

# model = mj.MjModel.from_xml_path('humanoid.xml')
# data = mj.MjData(model)

# Get noise
CTRL_STD = .05       # actuator units
# CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds
width = int(CTRL_RATE/model.opt.timestep)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
noise = util.FilteredNoise(model.nu, kernel, rng)
noisev = CTRL_STD * noise.sample(Tk-1)

### Get initial stabilizing controls (initial params for gradient descent)
util.reset(model, data, 10)
ctrls, K = opt_utils.get_stabilized_ctrls(model, data, Tk, noisev)
util.reset(model, data, 10)

### Gradient descent

qpos0 = data.qpos.copy()
# qs, qvels, ctrlsn = opt_utils.stabilized_grad_descent(
    # model, data, data.site('hand_right'), data.site('target'), ctrls, noisev)

# print()
# print(noisev[0])
# print()
# print('==========')
# print(qs[:3,:3])
# sys.exit()

while True:
    dlds = sites1 - sites2
    C = np.zeros((3, model.nv))
    util.reset(model, data, 10)
    qs, qvels = util.forward_sim(model, data, ctrls)
    mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)
    dldq = C.T @ dlds
    lams_fin = dldq

    losses = np.zeros(Tk)

    util.reset(model, data, 10)
    grads = opt_utils.traj_deriv(model, data, qs, qvels, ctrls,
                            lams_fin, losses, fixed_act_inds=other_a)
    ctrls[:,right_arm_a] = ctrls[:, right_arm_a] - 20*grads[:Tk-1]
    # env.reset(seed=seed)
    util.reset(model, data, 10)
    qpos0n = qpos0.copy()
    for k in range(Tk-1):
        if k % 10 == 0:
            qpos = data.qpos.copy()
            # data.qpos[:] = qpos0
            qpos0n[right_arm_j] = data.qpos[right_arm_j]
            data.qpos[:] = qpos0n
            ctrl0 = opt_utils.get_ctrl0(model, data)
            data.ctrl[:] = ctrl0
            K = opt_utils.get_feedback_ctrl_matrix(model, data)
            data.qpos[:] = qpos
        ctrl = opt_utils.get_lqr_ctrl_from_K(model, data, K, qpos0n, ctrl0)
        ctrl[right_arm_a] = ctrls[k, right_arm_a]
        print()
        print(k)
        print(ctrl)
        print()
        inp = ctrl + noisev[k]
        out = env.step(ctrl + noisev[k])
        observation, reward, terminated, __, info = out
        qs[k+1] = observation[:model.nq]
        qvels[k+1] = observation[model.nq:]

    print()
    print('==========')
    print(qs[:3,:3])
