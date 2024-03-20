import humanoid2d as h2d
# import baseball_lqr as lqr
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
    # render_mode='human',
    render_mode='rgb_array',
    frame_skip=1)
env.reset(seed=seed)
model = env.model
data = env.data

# Get noise
CTRL_STD = .05       # actuator units
# CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds
width = int(CTRL_RATE/model.opt.timestep)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
noise = util.FilteredNoise(model.nu, kernel, rng)
noisev = CTRL_STD * noise.sample(Tk-1)

### Get initial stabilizing controls
util.reset(model, data, 10)
# ctrls, K, __, __ = opt_utils.get_stabilized_ctrls(model, data, Tk, noisev,
                                          # data.qpos.copy())
ctrls, K = opt_utils.get_stabilized_simple(model, data, Tk, noisev)
print(ctrls[-5:])
util.reset(model, data, 10)
sys.exit()

# util.reset(model, data, 10)
# mj.mj_step1(model, data)
# data.ctrl[:] = ctrls[0] + noisev[0]
# mj.mj_step2(model, data)
# print(data.qpos)
# util.reset(model, data, 10)
# env.step(ctrls[0] + noisev[0])
# print(data.qpos)

qs, qvels = util.forward_sim(model, data, ctrls)
# print(qs[-3:,:3])
# sys.exit()
# util.reset(model, data, 10)
# ctrls, K = opt_utils.get_stabilized_ctrls(model, data, Tk, noisev=noisev)
# util.reset(model, data, 10)
# qs, qvels = util.forward_sim(model, data, ctrls)
# print(qs[-3:,:3])
# sys.exit()

util.reset(model, data, 10)

### Gradient descent

joints = opt_utils.get_joint_names(model)
right_arm_j = joints['right_arm_joint_inds']
right_arm_a = joints['right_arm_act_inds']
other_a = joints['non_right_arm_act_inds']

qpos0 = data.qpos.copy()

def get_losses(model, data, site1, site2):
    # I could put a forward sim here for safety (but less efficient)
    dlds = site1.xpos - site2.xpos
    C = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)
    dldq = C.T @ dlds
    lams_fin = dldq
    return np.zeros(Tk), lams_fin

# for k0 in range(3):
def get_stabilized_ctrls(model, data, Tk, noisev,
                        qpos0, free_act_ids=None, free_ctrls=None):
    if free_act_ids is None or len(free_act_ids) == 0:
        assert free_ctrls is None
        free_ctrls = np.empty((Tk, 0))
        free_act_ids = []
        free_jnt_ids = []
    else:
        free_jnt_ids = [model.actuator(k).trnid[0] for k in free_act_ids]
    qpos0n = qpos0.copy()
    qs = np.zeros((Tk, model.nq))
    qs[0] = data.qpos.copy()
    qvels = np.zeros((Tk, model.nq))
    qvels[0] = data.qvel.copy()
    for k in range(Tk-1):
        if k % 10 == 0:
            qpos0n[free_jnt_ids] = data.qpos[free_jnt_ids]
            ctrl0 = opt_utils.get_ctrl0(model, data, qpos0n)
            K = opt_utils.get_feedback_ctrl_matrix(model, data, ctrl0)
        ctrl = opt_utils.get_lqr_ctrl_from_K(model, data, K, qpos0n, ctrl0)
        ctrl[free_act_ids] = free_ctrls[k]
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrl + noisev[k]
        mj.mj_step2(model, data)
        qs[k+1] = data.qpos.copy()
        qvels[k+1] = data.qvel.copy()
    return qs, qvels

for k0 in range(3):
    lr = 20
    lams_fin = get_losses(model, data, data.site('hand_right'),
                          data.site('target'))[1]
    env.reset(seed=seed)
    util.reset(model, data, 10)
    grads = opt_utils.traj_deriv(model, data, qs, qvels, ctrls,
                            lams_fin, np.zeros(Tk), fixed_act_inds=other_a,
                                k0=k0)
    ctrls[:,right_arm_a] = ctrls[:, right_arm_a] - lr*grads[:Tk-1]

    __, __, qs, qvels = opt_utils.get_stabilized_ctrls(model, data, Tk,
                                     noisev, qpos0, 10, right_arm_a,
                                     ctrls[:, right_arm_a],)
print(qs[-3:,:3])

