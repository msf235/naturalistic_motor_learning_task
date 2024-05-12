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
    grab_targ = data.site('ball').xpos + np.array([0, 0, 0])
    s = sigmoid(5*np.linspace(0, 1, Tk1))
    s = np.tile(s, (3, 1)).T
    grab_traj = handx + s*(grab_targ - handx)
    # grab_traj[-1] = grab_targ

    setup_traj = np.zeros((Tk2, 3))
    s = np.linspace(0, 1, Tk2-Tk1)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])
    full_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs), axis=0)
    
    return full_traj

def sigmoid(x, a):
    # return .5 * (np.tanh(x-.5) + 1)
    return .5*np.tanh(a*(x-.5)) + .5

def tennis_traj(model, data, Tk):
    shouldxr = data.site('shoulder1_right').xpos
    shouldxl = data.site('shoulder1_left').xpos
    elbowx = data.site('elbow_right').xpos
    handxr = data.site('hand_right').xpos
    handxl = data.site('hand_left').xpos
    r1 = np.sum((shouldxr - elbowx)**2)**.5
    r2 = np.sum((elbowx - handxr)**2)**.5
    r = r1 + r2
    Tk1 = int(Tk / 3)
    Tk2 = int(1.2*Tk / 3)
    Tk3 = int(2.5*Tk/4)
    # Tk4 = int((Tk+Tk2)/2)

    # Right arm

    grab_targ = data.site('racket_handle').xpos + np.array([0, 0, -0.05])
    sx = np.linspace(0, 1, Tk1)
    # s = sigmoid(5*sx)
    s = sigmoid(sx, 5)
    # plt.plot(sx, s)
    # plt.show()
    # breakpoint()
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk2-Tk1, 3))), axis=0)
    grab_traj = handxr + s*(grab_targ - handxr)

    arc_traj_vs = arc_traj(data.site('shoulder1_right').xpos, r, np.pi,
                                  np.pi/2.5, Tk-Tk3, density_fn='')

    setup_traj = np.zeros((Tk3, 3))
    s = np.linspace(0, 1, Tk3-Tk2)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])

    right_arm_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs),
                                    axis=0)

    # plt.plot(right_arm_traj[:, 1], right_arm_traj[:, 2])
    # plt.scatter(shouldxr[1], shouldxr[2])
    # plt.axis('square')
    # plt.show()

    Tk1 = int(Tk / 3)
    Tk2 = int(1.2*Tk / 3)
    Tk3 = int(2*Tk/4)
    Tk4 = int(.8*Tk)
    # Tk4 = int((Tk+Tk2)/2)

    # Left arm
    grab_targ = data.site('ball').xpos + np.array([0, 0, 0])
    s = sigmoid(np.linspace(0, 1, Tk1), 5)
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk2-Tk1, 3))), axis=0)
    grab_traj = handxl + s*(grab_targ - handxl)

    arc_traj_vs = arc_traj(data.site('shoulder1_left').xpos, r,
                            0, .8*np.pi/2, Tk4-Tk3, density_fn='')
    arc_traj_vs2 = arc_traj(data.site('shoulder1_left').xpos, r,
                            .8*np.pi/2, .7*np.pi/2, Tk-Tk4, density_fn='')

    setup_traj = np.zeros((Tk3, 3))
    s = np.linspace(0, 1, Tk3-Tk2)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])

    left_arm_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs,
                                    arc_traj_vs2), axis=0)

    # plt.plot(left_arm_traj[:, 1], left_arm_traj[:, 2])
    # plt.scatter(shouldxl[1], shouldxl[2])
    # plt.axis('square')
    # plt.show()
    return right_arm_traj, left_arm_traj

def two_arm_idxs(model):
    two_arm_idx = {}
    joints = opt_utils.get_joint_ids(model)['body']
    acts = opt_utils.get_act_ids(model)

    body_j = joints['body_dofs']
    two_arm_idx['body_j'] = joints['body_dofs']
    arm_j = [k for k in body_j if k in joints['right_arm'] or k in
             joints['left_arm']]
    two_arm_idx['not_arm_j'] = [i for i in body_j if i not in arm_j]
    arm_a = [k for k in acts['all'] if k in acts['right_arm'] or
                           k in acts['left_arm']]
    two_arm_idx['not_arm_a'] = [k for k in acts['all'] if k not in arm_a and k
                               not in acts['adh']]
    two_arm_idx['right_arm_without_adh'] = [k for k in acts['right_arm'] if k
                                           not in acts['adh']]
    two_arm_idx['left_arm_without_adh'] = [k for k in acts['left_arm'] if k not
                                          in acts['adh']]
    return two_arm_idx

def one_arm_idxs(model, right_or_left='right'):
    joints = opt_utils.get_joint_ids(model)
    acts = opt_utils.get_act_ids(model)

    def ints(l1, l2):
        return list(set(l1).intersection(set(l2)))

    one_arm_idx = {}

    arm_j = joints['body'][f'{right_or_left}_arm']
    not_arm_j = [i for i in joints['body']['body_dofs'] if i not in arm_j]
    arm_a = acts[f'{right_or_left}_arm']
    arm_a_without_adh = [k for k in arm_a if k not in acts['adh']]
    # Include all adhesion (including other hand)
    arm_with_all_adh = [k for k in acts['all'] if k in arm_a or k in acts['adh']]
    not_arm_a = [k for k in acts['all'] if k not in arm_a and k not in
                 acts['adh']]
    one_arm_idx['arm_a_without_adh'] = arm_a_without_adh
    one_arm_idx['not_arm_j'] = not_arm_j
    one_arm_idx['not_arm_a'] = not_arm_a
    return one_arm_idx

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

def arm_target_traj(env, site_names, site_grad_idxs, stabilize_jnt_idx,
                    stabilize_act_idx, target_trajs, targ_traj_masks,
                    targ_traj_mask_types, ctrls, grad_trunc_tk, seed,
                    CTRL_RATE, CTRL_STD, Tk, max_its=30, lr=10, keep_top=1,
                    incr_every=5, amnt_to_incr=5):
    """Trains the right arm to follow the target trajectory (targ_traj). This
    involves gradient steps to update the arm controls and alternating with
    computing an LQR stabilizer to keep the rest of the body stable while the
    arm is moving.

    Args:
        site_names: list of site names
        site_grad_idxs: list of site gradient indices
        stabilize_jnt_idx: list of joint indices
        stabilize_act_idx: list of actuator indices
        target_trajs: list of target trajectories
        targ_traj_masks: list of target trajectory masks
        targ_traj_mask_types: list of target trajectory mask types
        ctrls: initial arm controls
        grad_trunc_tk: gradient truncation time
        seed: random seed
        CTRL_RATE: control rate
        CTRL_STD: control standard deviation
        Tk: number of time steps
        max_its: maximum number of gradient steps
        lr: learning rate
        keep_top: number of lowest losses to keep
        incr_every: number of gradient steps between increments
        amnt_to_incr: number of timesteps to increment the mask by each time
    """
    model = env.model
    data = env.data


    not_stabilize_act_idx = [k for k in range(model.nu) if k not in
                             stabilize_act_idx]

    n_sites = len(site_names)
    assert (n_sites == len(target_trajs) and n_sites == len(targ_traj_masks)
            and n_sites == len(targ_traj_mask_types))

    data0 = copy.deepcopy(data)


    noisev = make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE)

    qs, qvels = util.forward_sim(model, data, ctrls + noisev)
    util.reset_state(model, data, data0)

    ### Gradient descent
    qpos0 = data.qpos.copy()

    dt = model.opt.timestep
    T = Tk*dt
    tt = np.arange(0, T-dt, dt)

    optms = []
    targ_traj_progs = []
    targ_traj_mask_currs = []
    incr_cnts = []
    incr_everys = []
    amnt_to_incrs = []
    for k in range(n_sites):
        optms.append(opts.Adam(lr=lr))
        targ_traj_progs.append((isinstance(targ_traj_mask_types[k], str)
                                  and targ_traj_mask_types[k] == 'progressive'))
        targ_traj_mask_currs.append(targ_traj_masks[k])
        if targ_traj_progs[k]:
            targ_traj_mask_currs[k] = np.zeros((Tk,))
            incr_cnts.append(0)
    lowest_losses = LimLowestDict(keep_top)

    Tk1 = int(Tk / 3)
    for k0 in range(max_its):
        for k in range(n_sites):
            if targ_traj_progs[k] and k0 % incr_every == 0:
                idx = slice(amnt_to_incr*incr_cnts[k],
                            amnt_to_incr*(incr_cnts[k]+1))
                targ_traj_mask_currs[k][idx] = targ_traj_masks[k][idx]
                incr_cnts[k] += 1

        util.reset_state(model, data, data0)
        k, ball_contact = forward_to_contact(env, ctrls + noisev, render=False)
        util.reset_state(model, data, data0)
        grads = [0] * n_sites
        hxs = [0] * n_sites
        dldss = [0] * n_sites
        losses = [0] * n_sites
        for k in range(n_sites):
            grads[k], hxs[k], dldss[k] = opt_utils.traj_deriv_new(
                model, data, ctrls + noisev, target_trajs[k],
                targ_traj_mask_currs[k], grad_trunc_tk,
                deriv_ids=site_grad_idxs[k], deriv_site=site_names[k]
            )
            losses[k] = np.mean(dldss[k]**2)
            util.reset_state(model, data, data0)
        grads[0][:, :Tk1] *= 4
        for k in range(n_sites):
            # ctrls[:, right_arm_without_adh] = optm.update(
                # ctrls[:, right_arm_without_adh], grads1[:Tk-1], 'ctrls', loss1)
            ctrls[:, site_grad_idxs[k]] = optms[k].update(
                ctrls[:, site_grad_idxs[k]], grads[k][:Tk-1], 'ctrls',
                losses[k])

        ctrls, __, qs, qvels = opt_utils.get_stabilized_ctrls(
            model, data, Tk, noisev, qpos0, stabilize_act_idx,
            stabilize_jnt_idx, ctrls[:, not_stabilize_act_idx],
        )
        # ctrls, __, qs, qvels = opt_utils.get_stabilized_ctrls(
            # model, data, Tk, noisev, qpos0, not_arm_a,
            # not_arm_j, ctrls[:, arm_a],
        # )
        loss = sum([loss.item() for loss in losses]) / n_sites
        lowest_losses.append(loss, (k0, ctrls.copy()))
        print(loss)
        nr = range(n_sites)

        if k0 % incr_every == 0:
            fig, axs = plt.subplots(1, n_sites, figsize=(5*n_sites, 5))
            for k in nr:
                util.reset_state(model, data, data0)
                mj.mj_forward(model, data)
                hx = forward_with_site(env, ctrls, site_names[k], False)
                tm = np.tile(targ_traj_mask_currs[k], (3, 1)).T
                tm[tm == 0] = np.nan
                ft = target_trajs[k]*tm
                loss = np.mean((hx - target_trajs[k])**2)
                ax = axs[k]
                ax.plot(tt, hx[:,1], color='blue', label='x')
                ax.plot(tt, ft[:,1], '--', color='blue')
                ax.plot(tt, hx[:,2], color='red', label='y')
                ax.plot(tt, ft[:,2], '--', color='red')
                ax.set_title(site_names[k] + ' loss: ' + str(loss))
                ax.legend()
            fig.tight_layout()
            plt.show()

            util.reset_state(model, data, data0)
            qs, qvels = forward_to_contact(env, ctrls + noisev, True)

    return ctrls, lowest_losses.dict
