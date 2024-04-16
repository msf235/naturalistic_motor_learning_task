import humanoid2d as h2d
# import baseball_lqr as lqr
import opt_utils as opt_utils
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

def show_forward_sim(env, ctrls):
    for k in range(ctrls.shape[0]-1):
        # if k == 30:
            # breakpoint()
        util.step(env.model, env.data, ctrls[k])
        env.render()
        # env.step(ctrls[k])

def get_final_loss(model, data, xpos1, xpos2):
    # I could put a forward sim here for safety (but less efficient)
    dlds = xpos1 - xpos2
    C = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, C, None, site=data.site('hand_right').id)
    dldq = C.T @ dlds
    lams_fin = dldq # 11 and 12 are currently right shoulder and elbow
    loss = .5*np.mean(dlds**2)
    print(f'loss: {loss}', f'xpos1: {xpos1}', f'xpos2: {xpos2}')
    return loss, lams_fin

def forward_to_contact(env, ctrls, Tk, stop_on_contact=False):
    model = env.model
    data = env.data
    ball_contact = False
    for k in range(Tk-1):
        util.step(model, data, ctrls[k])
        env.render()
        contact_pairs = util.get_contact_pairs(model, data)
        if stop_on_contact:
            for cp in contact_pairs:
                if 'ball' in cp and 'hand_right' in cp:
                    ball_contact = True
                    break
    return k, ball_contact

def right_arm_target(env, target, body_pos, seed, CTRL_RATE, CTRL_STD,
                     Tk, stop_on_contact=False, target_name='ball',
                     max_its=30, lr=10):
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
    other_a = acts['non_right_arm']
    data0 = copy.deepcopy(data)
    noisev = make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

    ### Get initial stabilizing controls
    ctrls, K = opt_utils.get_stabilized_ctrls(
        model, data, Tk, noisev, data.qpos.copy(), non_adh, body_j
    )[:2]
    # ctrls[:, other_a] = ctrls_stab[:, other_a]
    util.reset_state(data, data0)
    qs, qvels = util.forward_sim(model, data, ctrls)
    util.reset_state(data, data0)

    ### Gradient descent
    qpos0 = data.qpos.copy()

    rhand = data.site('hand_right')

    ball_contact = False
    for k0 in range(max_its):
        loss, lams_fin = get_final_loss(model, data, rhand.xpos, target.xpos)
        # util.reset(model, data, 10, body_pos)
        util.reset_state(data, data0)
        # if k0 > 1:
        k, ball_contact = forward_to_contact(env, ctrls + noisev, Tk, stop_on_contact)
        # for k in range(Tk-1):
            # util.step(model, data, ctrls[k]+noisev[k])
            # env.render()
            # contact_pairs = util.get_contact_pairs(model, data)
            # for cp in contact_pairs:
                # if 'ball' in cp and 'hand_right' in cp:
                    # ball_contact = True
                    # break
        if ball_contact:
            break
        grads = opt_utils.traj_deriv(model, data, qs, qvels, ctrls, lams_fin,
                                     np.zeros(Tk), fixed_act_inds=other_a)
        ctrls[:, right_arm_a] = ctrls[:, right_arm_a] - lr*grads[:Tk-1]
        # qs, qvels = opt_utils.get_stabilized_ctrls(
            # model, data, Tk, noisev, qpos0, other_a, right_arm_a,
            # ctrls[:, right_arm_a]
        # )[2:] # should I update ctrls after this?
        __, __, qs, qvels = opt_utils.get_stabilized_ctrls(
            model, data, Tk, noisev, qpos0, other_a, not_right_arm_j,
            ctrls[:, right_arm_a]
        )
        if k0 > 100:
            breakpoint()
        print(ctrls[-5:,right_arm_a])
        print()
    return ctrls, k
