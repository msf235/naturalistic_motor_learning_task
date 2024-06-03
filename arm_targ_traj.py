import humanoid2d as h2d
import opt_utils as opt_utils
import optimizers as opts
import numpy as np
import sim_util as util
import mujoco as mj
import copy
import sortedcontainers as sc
from matplotlib import pyplot as plt

def make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE):
    acts = opt_utils.get_act_ids(model)
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

def tennis_traj(model, data, Tk):
    shouldx = data.site('shoulder1_right').xpos
    elbowx = data.site('elbow_right').xpos
    handx = data.site('hand_right').xpos
    r1 = np.sum((shouldx - elbowx)**2)**.5
    r2 = np.sum((elbowx - handx)**2)**.5
    r = r1 + r2
    Tk1 = int(Tk / 3)
    Tk2 = int(2*Tk/4)
    Tk3 = int((Tk+Tk2)/2)

    grab_targ = data.site('racket_handle').xpos + np.array([0, 0, -0.25])
    s = np.tanh(5*np.linspace(0, 1, Tk1))
    s = np.tile(s, (3, 1)).T
    grab_traj = handx + s*(grab_targ - handx)

    arc_traj_vs = arc_traj(data.site('shoulder1_right').xpos, r, np.pi,
                                  np.pi/2.5, Tk-Tk2-1, density_fn='')

    setup_traj = np.zeros((Tk2, 3))
    s = np.linspace(0, 1, Tk2-Tk1)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])

    # grab_traj[-1] = grab_targ

    full_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs), axis=0)

    # plt.plot(full_traj[:, 1], full_traj[:, 2])
    # plt.scatter(shouldx[1], shouldx[2])
    # plt.show()

    return full_traj

def show_forward_sim(env, ctrls):
    for k in range(ctrls.shape[0]-1):
        util.step(env.model, env.data, ctrls[k])
        env.render()

def forward_with_site(env, ctrls, site_name, render=False):
    site_xvs = np.zeros((ctrls.shape[0], 3))
    site_xvs[0] = env.data.site(site_name).xpos
    for k in range(ctrls.shape[0]-1):
        util.step(env.model, env.data, ctrls[k])
        site_xvs[k+1] = env.data.site(site_name).xpos
        if render:
            env.render()
    return site_xvs


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

class LimLowestDict:
    def __init__(self, max_len):
        self.max_len = max_len
        self.dict = sc.SortedDict()

    def append(self, key, val):
        self.dict.update({key: val})
        if len(self.dict) > self.max_len:
            self.dict.popitem()

def two_arm_target_traj(env,
                        target_traj1, targ_traj_mask1, targ_traj_mask_type1,
                        target_traj2, targ_traj_mask2, targ_traj_mask_type2,
                        ctrls, grad_trunc_tk, seed, CTRL_RATE, CTRL_STD, Tk,
                        max_its=30, lr=10, keep_top=1, incr_per1=5,
                        incr_per2=5):
    """Trains the right arm to follow the target trajectory (targ_traj). This
    involves gradient steps to update the arm controls and alternating with
    computing an LQR stabilizer to keep the rest of the body stable while the
    arm is moving."""
    model = env.model
    data = env.data

    data0 = copy.deepcopy(data)

    joints = opt_utils.get_joint_ids(model)['body']
    acts = opt_utils.get_act_ids(model)

    body_j = joints['body_dofs']
    arm_j = [k for k in body_j if k in joints['right_arm'] or k in
             joints['left_arm']]
    not_arm_j = [i for i in body_j if i not in arm_j]
    arm_a = [k for k in acts['all'] if k in acts['right_arm'] or k in
             acts['left_arm']]
    not_arm_a = [k for k in acts['all'] if k not in arm_a and k not in
                 acts['adh']]
    right_arm_without_adh = [k for k in acts['right_arm'] if k not in
                             acts['adh']]
    left_arm_without_adh = [k for k in acts['left_arm'] if k not in
                             acts['adh']]

    noisev = make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

    qs, qvels = util.forward_sim(model, data, ctrls + noisev)
    util.reset_state(data, data0)

    ### Gradient descent
    qpos0 = data.qpos.copy()

    dt = model.opt.timestep
    T = Tk*dt
    tt = np.arange(0, T-dt, dt)
    ball_contact = False
    optm = opts.Adam(lr=lr)
    optm2 = opts.Adam(lr=lr)
    targ_traj_prog1 = (isinstance(targ_traj_mask_type1, str)
                      and targ_traj_mask_type1 == 'progressive')
    targ_traj_mask_curr1 = targ_traj_mask1
    if targ_traj_prog1:
        targ_traj_mask_curr1 = np.zeros((Tk-1,))
        incr_cnt1 = 0
        amnt_to_incr1 = int(Tk / incr_per1) + 1
    targ_traj_prog2 = (isinstance(targ_traj_mask_type2, str)
                      and targ_traj_mask_type2 == 'progressive')
    targ_traj_mask_curr2 = targ_traj_mask2
    if targ_traj_prog2:
        targ_traj_mask_curr2 = np.zeros((Tk-1,))
        incr_cnt2 = 0
        amnt_to_incr2 = int(Tk / incr_per2) + 1


    lowest_losses = LimLowestDict(keep_top)

    Tk1 = int(Tk / 3)
    for k0 in range(max_its):
        if targ_traj_prog1 and k0 % incr_per1 == 0:
            idx = slice(amnt_to_incr1*incr_cnt1, amnt_to_incr1*(incr_cnt1+1))
            targ_traj_mask_curr1[idx] = targ_traj_mask1[idx]
            incr_cnt1 += 1
        if targ_traj_prog2 and k0 % incr_per2 == 0:
            idx = slice(amnt_to_incr2*incr_cnt2, amnt_to_incr2*(incr_cnt2+1))
            targ_traj_mask_curr2[idx] = targ_traj_mask2[idx]
            incr_cnt2 += 1

        util.reset_state(data, data0)
        k, ball_contact = forward_to_contact(env, ctrls + noisev, render=False)
        util.reset_state(data, data0)
        grads1, hxs1, dldss1 = opt_utils.traj_deriv(
            model, data, ctrls + noisev, target_traj1, targ_traj_mask1,
            grad_trunc_tk, deriv_ids=right_arm_without_adh,
            deriv_site='hand_right'
        )
        grads1[:Tk1] *= 20
        util.reset_state(data, data0)
        grads2, hxs2, dldss2 = opt_utils.traj_deriv(
            model, data, ctrls + noisev, target_traj2, targ_traj_mask2,
            grad_trunc_tk, deriv_ids=left_arm_without_adh,
            deriv_site='hand_left'
        )
        grads2[:Tk1] *= 20
        loss1 = np.mean(dldss1**2)
        ctrls[:, right_arm_without_adh] = optm.update(
            ctrls[:, right_arm_without_adh], grads1[:Tk-1], 'ctrls', loss1)
        loss2 = np.mean(dldss2**2)
        ctrls[:, left_arm_without_adh] = optm2.update(
            ctrls[:, left_arm_without_adh], grads2[:Tk-1], 'ctrls', loss2)

        util.reset_state(data, data0)
        ctrls, __, qs, qvels = opt_utils.get_stabilized_ctrls(
            model, data, Tk, noisev, qpos0, not_arm_a,
            not_arm_j, ctrls[:, arm_a],
        )
        loss = (loss1.item() + loss2.item()) / 2
        lowest_losses.append(loss, (k0, ctrls.copy()))
        print(loss)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    target_traj = target_traj1 * targ_traj_mask1.reshape(-1, 1)
    ax = axs[0]
    ax.plot(tt, hxs1[:,1], color='blue', label='x')
    ax.plot(tt, target_traj[:,1], '--', color='blue')
    ax.plot(tt, hxs1[:,2], color='red', label='y')
    ax.plot(tt, target_traj[:,2], '--', color='red')
    ax.set_title('Right hand')
    ax.legend()
    ax = axs[1]
    target_traj = target_traj2 * targ_traj_mask2.reshape(-1, 1)
    ax.plot(tt, hxs2[:,1], color='blue', label='x')
    ax.plot(tt, target_traj[:,1], '--', color='blue')
    ax.plot(tt, hxs2[:,2], color='red', label='y')
    ax.plot(tt, target_traj[:,2], '--', color='red')
    ax.set_title('Left hand')
    ax.legend()
    fig.tight_layout()
    plt.show()
    # util.reset_state(data, data0) # This is necessary, but why?
    # k, ball_contact = forward_to_contact(env, ctrls + noisev,
                                         # render=True)

    return ctrls, lowest_losses.dict

def arm_target_traj(env, target_traj, targ_traj_mask, targ_traj_mask_type,
                    ctrls, grad_trunc_tk, seed, CTRL_RATE, CTRL_STD, Tk,
                    max_its=30, lr=10, keep_top=1, right_or_left='right'):
    """Trains the right arm to follow the target trajectory (targ_traj). This
    involves gradient steps to update the arm controls and alternating with
    computing an LQR stabilizer to keep the rest of the body stable while the
    arm is moving."""
    model = env.model
    data = env.data

    data0 = copy.deepcopy(data)

    joints = opt_utils.get_joint_ids(model)
    acts = opt_utils.get_act_ids(model)

    def ints(l1, l2):
        return list(set(l1).intersection(set(l2)))

    arm_j = joints['body'][f'{right_or_left}_arm']
    not_arm_j = [i for i in joints['body']['body_dofs'] if i not in arm_j]
    arm_a = acts[f'{right_or_left}_arm']
    arm_a_without_adh = [k for k in arm_a if k not in acts['adh']]
    # Include all adhesion (including other hand)
    arm_with_all_adh = [k for k in acts['all'] if k in arm_a or k in acts['adh']]
    arm_with_all_adh.sort()
    not_arm_a = [k for k in acts['all'] if k not in arm_a and k not in
                 acts['adh']]

    deriv_site = f'hand_{right_or_left}'

    noisev = make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

    qs, qvels = util.forward_sim(model, data, ctrls)
    util.reset_state(data, data0)
    mj.mj_forward(model, data)
    print(data.site('hand_left').xpos, target_traj[0])
    print()

    ### Gradient descent
    qpos0 = data.qpos.copy()

    dt = model.opt.timestep
    T = Tk*dt
    tt = np.arange(0, T-dt, dt)
    ball_contact = False
    optm = opts.Adam(lr=lr)
    optm2 = opts.Adam(lr=lr)
    targ_traj_prog = (isinstance(targ_traj_mask_type, str)
                      and targ_traj_mask_type == 'progressive')
    targ_traj_mask_curr = targ_traj_mask
    if targ_traj_prog:
        targ_traj_mask_curr = np.zeros((Tk-1,))
        incr_per = 5 # increment period
        incr_cnt = 0

    lowest_losses = LimLowestDict(keep_top)

    Tk1 = int(Tk / 3)
    for k0 in range(max_its):
        if targ_traj_prog and k0 % incr_per == 0:
            idx = slice(10*incr_cnt, 10*(incr_cnt+1))
            targ_traj_mask_curr[idx] = targ_traj_mask[idx]
            incr_cnt += 1

        util.reset_state(data, data0)
        mj.mj_forward(model, data)
        # print(data.site('hand_left').xpos, target_traj[0])
        # print()
        k, ball_contact = forward_to_contact(env, ctrls + noisev,
                                             render=False)
        util.reset_state(data, data0)
        mj.mj_forward(model, data)
        grads1, hxs1, dldss1 = opt_utils.traj_deriv(
            model, data, ctrls + noisev, target_traj, targ_traj_mask,
            grad_trunc_tk, deriv_ids=arm_a_without_adh,
            deriv_site=deriv_site
        )
        loss1 = np.mean(dldss1**2)
        # util.reset_state(data, data0)
        # mj.mj_forward(model, data)
        # grads2, hxs2, dldss2 = opt_utils.traj_deriv(
            # model, data, ctrls + noisev, target_traj, targ_traj_mask,
            # grad_trunc_tk, deriv_ids=arm_a_without_adh,
            # deriv_site='ball_base'
        # )
        # print(data.site('hand_left').xpos, target_traj[0])
        # print()
        # breakpoint()
        # if k0 == max_its-1:
            # util.reset_state(data, data0)
            # hxs22 = forward_with_site(env, ctrls + noisev, 'ball_base',
                                      # render=True)
        # loss2 = .1*np.mean(dldss2**2)
        # loss = loss1 + loss2
        loss = loss1
        # grads = grads1 + .1*grads2
        grads = grads1
        ctrls[:, arm_a_without_adh] = optm.update(
            ctrls[:, arm_a_without_adh], grads, 'ctrls', loss)
        # if k0 == max_its-1:
            # util.reset_state(data, data0)
            # hxs23 = forward_with_site(env, ctrls + noisev, 'ball_base',
                                      # render=True)
            # breakpoint()
        util.reset_state(data, data0)
        mj.mj_forward(model, data)
        ctrls, __, qs, qvels = opt_utils.get_stabilized_ctrls(
            model, data, Tk, noisev, qpos0,
            not_arm_a,
            not_arm_j, ctrls[:, arm_with_all_adh],
        )
        lowest_losses.append(loss.item(), (k0, ctrls.copy()))
        print(loss.item())
        util.reset_state(data, data0)
        mj.mj_forward(model, data)
        # hxs1 = forward_with_site(env, ctrls, 'hand_right', False)
        # loss1 = np.mean((hxs1 - target_traj)**2)
        # util.reset_state(data, data0)
        # hxs2 = forward_with_site(env, ctrls, 'ball_base', False)
        # loss2 = .1*np.mean((hxs2 - target_traj)**2)
        # loss = loss1 + loss2
        # lowest_losses.append(loss, (k0, ctrls.copy()))
        # print(list(lowest_losses.dict.keys()))
        # print(data.site('hand_left').xpos)
        # print()

        # fig, ax = plt.subplots()
        # target_traj = target_traj * targ_traj_mask.reshape(-1, 1)
        # ax.plot(tt, hxs1[:,1], color='blue', label='x')
        # # ax.plot(tt, hxs2[:,1], '-.', color='blue', label='x_ball')
        # ax.plot(tt, target_traj[:,1], '--', color='blue', label='x_targ')
        # ax.plot(tt, hxs1[:,2], color='red', label='y')
        # # ax.plot(tt, hxs2[:,2], '-.', color='red', label='y_ball')
        # ax.plot(tt, target_traj[:,1], '--', color='blue')
        # ax.plot(tt, target_traj[:,2], '--', color='red')
        # ax.legend()
        # plt.show()
    util.reset_state(data, data0) # This is necessary, but why?
    mj.mj_forward(model, data)
    k, ball_contact = forward_to_contact(env, ctrls + noisev,
                                         render=True)
    # print(hxs2[-5:,:5])

    return ctrls, lowest_losses.dict

