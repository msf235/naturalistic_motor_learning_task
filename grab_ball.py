import humanoid2d as h2d
# import baseball_lqr as lqr
import opt_utils as opt_utils
import optimizers as opts
import numpy as np
import sim_util as util
import mujoco as mj
import sys
import os
import copy
import time
import pickle as pkl
from matplotlib import pyplot as plt

def make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE):
    acts = opt_utils.get_act_names(model)
    adh = acts['adh_right_hand']
    rng = np.random.default_rng(seed)
    width = int(CTRL_RATE/model.opt.timestep)
    kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
    kernel /= np.linalg.norm(kernel)
    noise = util.FilteredNoise(model.nu, kernel, rng)
    noisev = CTRL_STD * noise.sample(Tk-1)
    noisev[:, adh] = 0
    return noisev

def arc_traj(x0, r, theta0, theta1, n, density_fn='uniform'):
    if density_fn != 'uniform':
        unif = np.linspace(0, 1, n)
        theta = (theta1-theta0)*unif**1.5 + theta0
    else:
        theta = np.linspace(theta0, theta1, n)

    x = x0 + r*np.array([0*theta, np.cos(theta), np.sin(theta)]).T
    return x

def throw_traj(model, data, Tk):
    shouldx = data.site('shoulder1_right').xpos
    elbowx = data.site('elbow_right').xpos
    handx = data.site('hand_right').xpos
    ballx = data.site('ball_base').xpos
    r1 = np.sum((shouldx - elbowx)**2)**.5
    r2 = np.sum((elbowx - handx)**2)**.5
    r = r1 + r2
    Tk1 = int(Tk / 3)
    Tk2 = int(2*Tk/4)
    Tk3 = int((Tk+Tk2)/2)
    arc_traj_vs = arc_traj(data.site('shoulder1_right').xpos, r, np.pi,
                                  np.pi/2.5, Tk-Tk2-1, density_fn='')
    grab_targ = data.site('ball_base').xpos + np.array([0, 0, -0.01])
    s = np.tanh(5*np.linspace(0, 1, Tk1))
    s = np.tile(s, (3, 1)).T
    grab_traj = handx + s*(grab_targ - handx)
    # grab_traj[-1] = grab_targ

    setup_traj = np.zeros((Tk2, 3))
    s = np.linspace(0, 1, Tk2-Tk1)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])
    full_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs), axis=0)
    
    return full_traj

def show_forward_sim(env, ctrls):
    for k in range(ctrls.shape[0]-1):
        util.step(env.model, env.data, ctrls[k])
        env.render()

def forward_to_contact(env, ctrls, render=True):
    model = env.model
    data = env.data
    ball_contact = False
    Tk = ctrls.shape[0]
    for k in range(Tk):
        util.step(model, data, ctrls[k])
        if render:
            env.render()
        contact_pairs = util.get_contact_pairs(model, data)
        for cp in contact_pairs:
            if 'ball' in cp and 'hand_right' in cp:
                ball_contact = True
    return k, ball_contact

def right_arm_target_traj(env, target_traj, targ_traj_mask,
                          targ_traj_mask_type, ctrls,
                          grad_trunc_tk, seed, CTRL_RATE, CTRL_STD, Tk,
                          max_its=30, lr=10):
    """Trains the right arm to follow the target trajectory (targ_traj). This
    involves gradient steps to update the arm controls and alternating with
    computing an LQR stabilizer to keep the rest of the body stable while the
    arm is moving."""
    model = env.model
    data = env.data

    data0 = copy.deepcopy(data)

    joints = opt_utils.get_joint_names(model)
    right_arm_j = joints['right_arm']
    body_j = joints['body']
    not_right_arm_j = [i for i in body_j if i not in right_arm_j]
    acts = opt_utils.get_act_names(model)
    right_arm_a_without_adh = acts['right_arm']
    right_arm_a = acts['right_arm_with_adh']
    adh = acts['adh_right_hand']
    non_adh = acts['non_adh']
    not_right_arm_a = acts['non_right_arm']
    not_right_arm_a_not_adh = acts['non_right_arm_non_adh']
    noisev = make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

    qs, qvels = util.forward_sim(model, data, ctrls)
    util.reset_state(data, data0)

    ### Gradient descent
    qpos0 = data.qpos.copy()

    rhand = data.site('hand_right')

    dt = model.opt.timestep
    T = Tk*dt
    tt = np.arange(0, T-dt, dt)
    ball_contact = False
    optm = opts.Adam(lr=lr)
    targ_traj_prog = (isinstance(targ_traj_mask_type, str)
                      and targ_traj_mask_type == 'progressive')
    targ_traj_mask_curr = targ_traj_mask
    if targ_traj_prog:
        targ_traj_mask_curr = np.zeros((Tk-1,))
        incr_per = 5 # increment period
        incr_cnt = 0

    Tk1 = int(Tk / 3)
    for k0 in range(max_its):
        if targ_traj_prog and k0 % incr_per == 0:
            idx = slice(10*incr_cnt, 10*(incr_cnt+1))
            targ_traj_mask_curr[idx] = targ_traj_mask[idx]
            incr_cnt += 1

        util.reset_state(data, data0)
        k, ball_contact = forward_to_contact(env, ctrls + noisev,
                                             render=False)
        util.reset_state(data, data0)
        grads, hxs, dldss = opt_utils.traj_deriv(model, data, ctrls + noisev,
                                          target_traj, targ_traj_mask,
                                          grad_trunc_tk,
                                          fixed_act_inds=not_right_arm_a)
        # grads[:Tk1] = 2*grads[:Tk1]
        loss = np.mean(dldss**2)
        ctrls[:, right_arm_a_without_adh] = optm.update(
            ctrls[:, right_arm_a_without_adh], grads[:Tk-1], 'ctrls', loss)
        # ctrls[:, right_arm_a] = ctrls[:, right_arm_a] - lr*grads[:Tk-1]
        util.reset_state(data, data0) # This is necessary, but why?
        ctrls, __, qs, qvels = opt_utils.get_stabilized_ctrls(
            model, data, Tk, noisev, qpos0, not_right_arm_a_not_adh,
            not_right_arm_j, ctrls[:, right_arm_a]
        )
        print(loss.item())
    fig, ax = plt.subplots()
    target_traj = target_traj * targ_traj_mask.reshape(-1, 1)
    ax.plot(tt, hxs[:,1], color='blue')
    ax.plot(tt, target_traj[:,1], '--', color='blue')
    ax.plot(tt, hxs[:,2], color='red')
    ax.plot(tt, target_traj[:,2], '--', color='red')
    plt.show()
    util.reset_state(data, data0) # This is necessary, but why?
    k, ball_contact = forward_to_contact(env, ctrls + noisev,
                                         render=True)
    breakpoint()

    return ctrls

