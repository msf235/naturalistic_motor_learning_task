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

def arc_traj(x0, r, theta0, theta1, n):
    theta = np.linspace(theta0, theta1, n)
    x = x0 + r*np.array([0*theta, np.cos(theta), np.sin(theta)]).T
    return x

def show_forward_sim(env, ctrls):
    for k in range(ctrls.shape[0]-1):
        # if k == 30:
            # breakpoint()
        util.step(env.model, env.data, ctrls[k])
        env.render()
        # env.step(ctrls[k])

def get_final_loss(model, data, xpos1, xpos2):
    # I could put a forward sim here for safety (but less efficient)
    # mj.mj_forward(model, data)
    dlds = xpos1 - xpos2
    C = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)
    dldq = C.T @ dlds
    lams_fin = dldq # 11 and 12 are currently right shoulder and elbow
    loss = .5*np.mean(dlds**2)
    print(f'loss: {loss}', f'xpos1: {xpos1}', f'xpos2: {xpos2}')
    return loss, lams_fin

def forward_to_contact(env, ctrls, stop_on_contact=False, render=True):
    model = env.model
    data = env.data
    ball_contact = False
    Tk = ctrls.shape[0]
    for k in range(Tk):
        util.step(model, data, ctrls[k])
        if render:
            env.render()
        contact_pairs = util.get_contact_pairs(model, data)
        if stop_on_contact:
            for cp in contact_pairs:
                if 'ball' in cp and 'hand_right' in cp:
                    pass
                    # breakpoint()
                    # ctrls[adh] = 1
                    # ball_contact = True
                    # break
    return k, ball_contact

def right_arm_target_traj(env, target_traj, targ_traj_mask, ctrls,
                          grad_trunc_tk, seed, CTRL_RATE, CTRL_STD, Tk,
                          stop_on_contact=False, target_name='ball',
                          max_its=30, lr=10):
    model = env.model
    data = env.data

    joints = opt_utils.get_joint_names(model)
    right_arm_j = joints['right_arm']
    body_j = joints['body']
    not_right_arm_j = [i for i in body_j if i not in right_arm_j]
    acts = opt_utils.get_act_names(model)
    # right_arm_a = acts['right_arm']
    right_arm_a = acts['right_arm_with_adh']
    adh = acts['adh_right_hand']
    non_adh = acts['non_adh']
    not_right_arm_a = acts['non_right_arm']
    data0 = copy.deepcopy(data)
    noisev = make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

    qs, qvels = util.forward_sim(model, data, ctrls)
    util.reset_state(data, data0)

    ### Gradient descent
    qpos0 = data.qpos.copy()

    rhand = data.site('hand_right')

    from matplotlib import pyplot as plt
    dt = model.opt.timestep
    T = Tk*dt
    tt = np.arange(0, T-dt, dt)
    ball_contact = False
    optm = opts.Adam(lr=lr)
    targ_traj_mask_bool = False
    if isinstance(targ_traj_mask, str) and targ_traj_mask == 'progressive':
        targ_traj_mask_bool = True
        targ_traj_mask = np.zeros((Tk-1,))
        incr_per = 5 # increment period
        incr_cnt = 0

    for k0 in range(max_its):
        if targ_traj_mask_bool and k0 % incr_per == 0:
            targ_traj_mask[10*incr_cnt:10*(incr_cnt+1)] = 1
            incr_cnt += 1

        util.reset_state(data, data0)
        k, ball_contact = forward_to_contact(env, ctrls + noisev,
                                             stop_on_contact, render=False)
        # if ball_contact:
            # ctrls[adh] = 1
        util.reset_state(data, data0)
        grads, hxs, dldss = opt_utils.traj_deriv(model, data, ctrls + noisev,
                                          target_traj, targ_traj_mask,
                                          grad_trunc_tk,
                                          fixed_act_inds=not_right_arm_a)
        loss = np.mean(dldss**2)
        ctrls[:, right_arm_a] = optm.update(ctrls[:, right_arm_a],
                                            grads[:Tk-1], 'ctrls', loss)
        # ctrls[:, right_arm_a] = ctrls[:, right_arm_a] - lr*grads[:Tk-1]
        util.reset_state(data, data0) # This is necessary, but why?
        ctrls, __, qs, qvels = opt_utils.get_stabilized_ctrls(
            model, data, Tk, noisev, qpos0, not_right_arm_a, not_right_arm_j,
            ctrls[:, right_arm_a]
        )
        # ctrls[adh] = 1
    util.reset_state(data, data0) # This is necessary, but why?
    fig, ax = plt.subplots()
    # ax.axis('square')
    target_traj = target_traj * targ_traj_mask.reshape(-1, 1)
    ax.plot(tt, hxs[:,1], color='blue')
    ax.plot(tt, target_traj[:,1], '--', color='blue')
    ax.plot(tt, hxs[:,2], color='red')
    ax.plot(tt, target_traj[:,2], '--', color='red')
    plt.show()
    forward_to_contact(env, ctrls, stop_on_contact, True)
    return ctrls, k

