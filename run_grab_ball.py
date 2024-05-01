import humanoid2d as h2d
import opt_utils as opt_utils
import numpy as np
import sim_util as util
import mujoco as mj
import sys
import os
import pickle as pkl
import arm_targ_traj as arm_t
from matplotlib import pyplot as plt

### Set things up
seed = 2
out_f = 'grab_ball_ctrl.np'
# rerun = False
rerun = True

# Tk = 500
# Tk = 200
# Tk = 100
# Tk = 80
Tk = 120
# Tk = 200

body_pos = -0.3

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    # "distance": 4.0,
    "distance": 5,
    "lookat": np.array((0.0, 0.0, 1.15)),
    # "elevation": -20.0,
    "elevation": -10.0,
    "azimuth": 180,
}
# Create a Humanoid2dEnv object
env = h2d.Humanoid2dEnv(
    render_mode='human',
    # render_mode='rgb_array',
    frame_skip=1,
    default_camera_config=DEFAULT_CAMERA_CONFIG)
model = env.model
data = env.data

joints = opt_utils.get_joint_names(model)
right_arm_j = joints['right_arm']
body_j = joints['body']
not_right_arm_j = [i for i in body_j if i not in right_arm_j]
acts = opt_utils.get_act_names(model)
right_arm_a = acts['right_arm']
adh = acts['adh_right_hand']
non_adh = acts['non_adh']
non_right_arm_a = acts['non_right_arm']
all_act_ids = list(range(model.nu))

env.reset(seed=seed) # necessary?

util.reset(model, data, 10, body_pos)

# show_forward_sim(model, data, np.zeros((Tk, model.nu)))

# Get noise
# CTRL_STD = .05       # actuator units
CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds

# if rerun or not os.path.exists(out_f):
# throw_target = data.site('target2')
target = data.site('ball_base')
# target = data.site('target2')

shouldx = data.site('shoulder1_right').xpos
elbowx = data.site('elbow_right').xpos
handx = data.site('hand_right').xpos
ballx = data.site('ball_base').xpos
r1 = shouldx - elbowx
r1 = np.sum(r1**2)**.5
r2 = elbowx - handx
r2 = np.sum(r2**2)**.5
r = (r1 + r2)

# Tk1 = 50
# Tk1 = Tk // 4
Tk1 = int(Tk / 3)
# Tk2 = int(3*Tk/4)
Tk2 = int(2*Tk/4)
Tk3 = int((Tk+Tk2)/2)
# Tk2 = int(3*Tk / 4)
# Get angle between ball and shoulder
arc_traj = arm_t.arc_traj(data.site('shoulder1_right').xpos, r, np.pi,
                              np.pi/2.5, Tk-Tk2-1, density_fn='')
# from matplotlib import pyplot as plt
# plt.plot(arc_traj[:,1], arc_traj[:,2], '-x')
# plt.axis('square')
# plt.show()
# sys.exit()
grab_targ = data.site('ball_base').xpos
s = np.tanh(5*np.linspace(0, 1, Tk1))
s = np.tile(s, (3, 1)).T
# from matplotlib import pyplot as plt
# plt.plot(s[:,0])
# plt.show()
grab_traj = handx + s*(grab_targ - handx)

setup_traj = np.zeros((Tk2, 3))
s = np.linspace(0, 1, Tk2-Tk1)
s = np.stack((s, s, s)).T
setup_traj = grab_traj[-1] + s*(arc_traj[0] - grab_traj[-1])
# grab_traj += handx
full_traj = np.concatenate((grab_traj, setup_traj, arc_traj), axis=0)
targ_traj_mask = np.ones((Tk-1,))
# targ_traj_mask[Tk2:] = 0
# targ_traj_mask[-1] = 1
# targ_traj_mask[Tk3] = 1
# targ_traj_mask[Tk3] = 1
# targ_traj_mask[Tk2:] = 0

targ_traj_mask_type = 'progressive'


# lr = .1/Tk
# lr = 1/Tk
lr = 2/Tk
# lr = .001
# lr = .01
# lr = .002

max_its = 100
# max_its = 50

# full_traj = grab_traj
# Tk = Tk1
# full_traj = np.zeros((Tk-1, 3))
# util.reset(model, data, 10, body_pos)
# mj.mj_forward(model, data)
# full_traj[:10] = data.site('hand_right').xpos
# full_traj[30:40] = data.site('hand_right').xpos - np.array((0, .1, -.1))
# full_traj[:] = data.site('hand_right').xpos - np.array((0, .1, -.1))
# full_traj[-5:] = data.site('target').xpos
# targ_traj_mask = np.zeros((Tk-1,))
# targ_traj_mask[-20:-10] = 1
# targ_traj_mask[:20] = 1
# targ_traj_mask[30:40] = 1
# targ_traj_mask[-5:] = 1

# plt.plot(arc_traj[:,1], arc_traj[:,2])
# plt.plot(full_traj[:,1], full_traj[:,2])
# plt.axis('square')
# plt.show()
# target_traj = arm_t.arc_traj(model.joint('shoulder1_right'), 

noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)


if rerun or not os.path.exists(out_f):
    ### Get initial stabilizing controls
    util.reset(model, data, 10, body_pos)
    adhs = np.ones((Tk, 1))
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), non_adh, body_j,
        free_ctrls=adhs)[:2]
    util.reset(model, data, 10, body_pos)
    # k, ball_contact = arm_t.forward_to_contact(env, ctrls + noisev,
                                         # False, render=True)
    # ctrls[adh] = 1
    ctrls, k = arm_t.right_arm_target_traj(
        env, full_traj, targ_traj_mask, targ_traj_mask_type, ctrls, 30, seed,
        CTRL_RATE, CTRL_STD, Tk,
        stop_on_contact=True,
        # stop_on_contact=False,
        lr=lr, max_its=max_its)
    with open(out_f, 'wb') as f:
        pkl.dump({'ctrls': ctrls, 'k': k}, f)
else:
    with open(out_f, 'rb') as f:
        out = pkl.load(f)
        ctrls = out['ctrls']
        k = out['k']


noisev = arm_t.make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

util.reset(model, data, 10, body_pos)
ctrls_n = (ctrls+noisev)[:k+1]
arm_t.show_forward_sim(env, ctrls_n)

breakpoint()

ctrls, k = arm_t.right_arm_target(env, throw_target, ctrls_n,
                                      seed, CTRL_RATE, CTRL_STD, 10*Tk,
                                      lr=1, stop_on_contact=False)

sys.exit()







# ctrl = ctrls[k]
# ctrl[adh] = 1
# ctrl[right_arm_a] = -.1
fact = -.7
throw_ctrls = np.zeros((Tk-1, 2))
throw_ctrls[:, 0] = fact
throw_ctrls[:, 1] = fact
qpos0 = data.qpos.copy()
ctrls2 = opt_utils.get_stabilized_ctrls(
    model, data, Tk, noisev, qpos0, non_right_arm_a, not_right_arm_j,
    throw_ctrls
)[0]
ctrls2[:,adh]=1
util.reset(model, data, 10, body_pos)
arm_t.show_forward_sim(env, ctrls_n)
arm_t.show_forward_sim(env, ctrls2)

breakpoint()

throw_targ = [-0.2, 0.4, 0]

# breakpoint()
