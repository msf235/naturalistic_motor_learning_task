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
env = h2d.Humanoid2dEnv(render_mode='human', frame_skip=1)
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
# ctrls, K = opt_utils.get_stabilized_ctrls(model, data, Tk, noisev)
ctrls, K = opt_utils.get_stabilized_simple(model, data, Tk, noisev)
util.reset(model, data, 10)

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
def get_stabilized_ctrls(model, data, right_arm_a, Tk, noisev,
                        qpos0, ctrls, k0):
    # util.reset(model, data, 10)
    # mj.mj_step1(model, data)
    # data.ctrl[:] = ctrls[0] + noisev[0]
    # mj.mj_step2(model, data)
    # print(data.qpos)
    # util.reset(model, data, 10)
    # env.step(ctrls[0] + noisev[0])
    # print(data.qpos)
    # sys.exit()
    right_arm_j = [model.actuator(k).trnid[0] for k in right_arm_a]
    qpos0n = qpos0.copy()
    # qs = np.zeros((Tk, model.nq))
    # qs[0] = data.qpos.copy()
    # qvels = np.zeros((Tk, model.nq))
    # qvels[0] = data.qvel.copy()
    for k in range(Tk-1):
        if k % 10 == 0:
            qpos0n[right_arm_j] = data.qpos[right_arm_j]
            ctrl0 = opt_utils.get_ctrl0(model, data, qpos0n)
            K = opt_utils.get_feedback_ctrl_matrix(model, data, ctrl0)
        ctrl = opt_utils.get_lqr_ctrl_from_K(model, data, K, qpos0n, ctrl0)
        ctrl[right_arm_a] = ctrls[k]
        # if k0 == 1: # ctrls is different
            # breakpoint()
        out = env.step(ctrl + noisev[k])
        qs[k+1] = data.qpos.copy()
        qvels[k+1] = data.qvel.copy()
        # observation, reward, terminated, __, info = out
        # qs[k+1] = observation[:model.nq]
        # qvels[k+1] = observation[model.nq:]
        # if k0 == 1:
            # breakpoint()
    print(qs[-3:,:3])
    print(qvels[-3:,:3])
    # return qs
    return qs, qvels

for k0 in range(2):
    lr = 20
    lams_fin = get_losses(model, data, data.site('hand_right'),
                          data.site('target'))[1]
    env.reset(seed=seed)
    util.reset(model, data, 10)
    if k0 == 1:
        print(ctrls[:2])
    grads = opt_utils.traj_deriv(model, data, qs, qvels, ctrls,
                            lams_fin, np.zeros(Tk), fixed_act_inds=other_a,
                                k0=k0)
    if k0 == 1:
        print(grads[:2])
        breakpoint()
    ctrls[:,right_arm_a] = ctrls[:, right_arm_a] - lr*grads[:Tk-1]

    # ctrls, K, qs, qvels = opt_utils.get_stabilized_ctrls(
        # model, data, Tk, noisev, right_arm_a, ctrls, 10)
    # get_stabilized_ctrls(model, data, right_arm_j, right_arm_a, Tk,
                              # noisev, qpos0)
    qs, qvels = get_stabilized_ctrls(model, data, right_arm_a, Tk,
                                     noisev, qpos0,
                                     ctrls[:, right_arm_a],
                                     k0
                                     # ctrls
                                    )
# print()
# print(qs[-3:,:3])

