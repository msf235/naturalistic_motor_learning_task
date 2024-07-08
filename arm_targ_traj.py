import humanoid2d as h2d
import opt_utils as opt_utils
import optimizers as opts
import numpy as np
import sim_util as util
import mujoco as mj
import copy
import sortedcontainers as sc
from matplotlib import pyplot as plt
import time
import basic_movements

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

def sigmoid(x, a):
    # return .5 * (np.tanh(x-.5) + 1)
    return .5*np.tanh(a*(x-.5)) + .5

def throw_grab_traj(model, data, Tk):
    shouldx = data.site('shoulder1_right').xpos
    elbowx = data.site('elbow_right').xpos
    handx = data.site('hand_right').xpos
    r1 = np.sum((shouldx - elbowx)**2)**.5
    r2 = np.sum((elbowx - handx)**2)**.5
    r = r1 + r2
    Tk1 = int(Tk / 3)
    # Tk2 = int(2*Tk/3)
    Tk2 = Tk - Tk1
    Tk3 = int((Tk+Tk2)/2)
    arc_traj_vs = arc_traj(data.site('shoulder1_right').xpos, r, np.pi,
                                  np.pi/2.2, Tk-Tk2, density_fn='')
    grab_targ = data.site('ball').xpos + np.array([0, 0, 0])
    s = sigmoid(np.linspace(0, 1, Tk1), 5)
    s = np.tile(s, (3, 1)).T
    grab_traj = handx + s*(grab_targ - handx)
    # grab_traj[-1] = grab_targ

    setup_traj = np.zeros((Tk2, 3))
    s = np.linspace(0, 1, Tk2)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])
    full_traj = np.concatenate((grab_traj, setup_traj), axis=0)

    time_dict = {
        't_1': Tk1,
        't_2': Tk2,
        't_3': Tk3,
        'Tk1': Tk1,
        'Tk2': Tk2-Tk1,
        'Tk3': Tk3-Tk2
    }
    
    return full_traj, time_dict

def throw_traj(model, data, Tk):
    shouldx = data.site('shoulder1_right').xpos
    elbowx = data.site('elbow_right').xpos
    handx = data.site('hand_right').xpos
    r1 = np.sum((shouldx - elbowx)**2)**.5
    r2 = np.sum((elbowx - handx)**2)**.5
    r = r1 + r2
    Tk1 = int(Tk / 3)
    Tk2 = int(2*Tk/3)
    Tk3 = int((Tk+Tk2)/2)
    arc_traj_vs = arc_traj(data.site('shoulder1_right').xpos, r, np.pi,
                                  np.pi/2.2, Tk-Tk2, density_fn='')
    grab_targ = data.site('ball').xpos + np.array([0, 0, 0])
    s = sigmoid(np.linspace(0, 1, Tk1), 5)
    s = np.tile(s, (3, 1)).T
    grab_traj = handx + s*(grab_targ - handx)
    # grab_traj[-1] = grab_targ

    setup_traj = np.zeros((Tk2, 3))
    s = np.linspace(0, 1, Tk2-Tk1)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])
    full_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs), axis=0)

    time_dict = {
        't_1': Tk1,
        't_2': Tk2,
        't_3': Tk3,
        'Tk1': Tk1,
        'Tk2': Tk2-Tk1,
        'Tk3': Tk3-Tk2
    }
    
    return full_traj, time_dict


def tennis_grab_traj(model, data, Tk):
    shouldxr = data.site('shoulder1_right').xpos
    shouldxl = data.site('shoulder1_left').xpos
    elbowx = data.site('elbow_right').xpos
    handxr = data.site('hand_right').xpos
    handxl = data.site('hand_left').xpos
    r1 = np.sum((shouldxr - elbowx)**2)**.5
    r2 = np.sum((elbowx - handxr)**2)**.5
    r = r1 + r2
    Tk_right_1 = int(Tk / 4) # Time to grab with right hand (1)
    Tk_right_2 = int(Tk / 12) # Time to grab with right hand (2)
    t_right_1 = Tk_right_1 + Tk_right_2
    Tk_right_3 = Tk - t_right_1

    Tk_left_1 = int(Tk / 3) # Duration to grab with left hand (1)
    Tk_left_2 = int(Tk / 8) # Duration to grab with left hand (2)
    t_left_1 = Tk_left_1 + Tk_left_2 # Time up to end of grab
    Tk_left_3 = Tk - t_left_1 # Duration to set up

    # Tk4 = int((Tk+Tk2)/2)

    # fig, ax = plt.subplots()
    # tt = np.linspace(0, 1, Tk)

    # Right arm

    # grab_targ = data.site('racket_handle').xpos + np.array([0, 0, -0.05])
    grab_targ = data.site('racket_handle_top').xpos + np.array([0, 0, 0.03])
    # grab_targ = data.site('racket_handle_top').xpos + np.array([0, 0, 0])
    sx = np.linspace(0, 1, Tk_right_1)
    s = sigmoid(sx, 5)
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk_right_2, 3))), axis=0)
    grab_traj = handxr + s*(grab_targ - handxr)

    arc_traj_vs = arc_traj(data.site('shoulder1_right').xpos, r, np.pi,
                                  np.pi/6, 10, density_fn='')

    s = np.linspace(0, 1, Tk_right_3)
    s = sigmoid(s, 5)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])

    right_arm_traj = np.concatenate((grab_traj, setup_traj),
                                    axis=0)

    t_fin = Tk * model.opt.timestep
    tt = np.linspace(0, t_fin, Tk)

    # fig, ax = plt.subplots()
    # ax.plot(tt[:t_right_1], grab_traj[:, 2], c='blue')
    # ax.plot(tt[t_right_1:t_right_2], setup_traj[:, 2], c='red')
    # ax.plot(tt[t_right_2:], arc_traj_vs[:, 2], c='blue')
    # plt.show()

    # Tk4 = int((Tk+Tk2)/2)

    # Left arm
    # grab_targ = data.site('ball').xpos + np.array([0, 0, 0.04])
    grab_targ = data.site('ball_top').xpos + np.array([0, 0, .03])
    s = sigmoid(np.linspace(0, 1, Tk_left_1), 5)
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk_left_2, 3))), axis=0)
    grab_traj = handxl + s*(grab_targ - handxl)

    arc_traj_vs = arc_traj(data.site('shoulder1_left').xpos, r,
                            np.pi/5, np.pi/2, 10, density_fn='')
    xs = arc_traj_vs[:, 1].copy()
    x0 = xs[0]
    recenter_scale_xs = .8*(xs - x0)
    arc_traj_vs[:,1] = recenter_scale_xs + x0
    # arc_traj_vs2 = arc_traj(data.site('shoulder1_left').xpos, r,
                            # .9*np.pi/2, .7*np.pi/2, Tk_left_5, density_fn='')
    # arc_traj_vs2 = arc_traj_vs[:-Tk_left_5:-1]
    arc_traj_vs2 = arc_traj(data.site('shoulder1_left').xpos, r,
                            .9*np.pi/2, .7*np.pi/2, 10, density_fn='')

    setup_traj = np.zeros((Tk_left_3, 3))
    s = np.linspace(0, 1, Tk_left_3)
    s = sigmoid(s, 5)
    # s = 2*sigmoid(.5*s, 5)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])

    left_arm_traj = np.concatenate((grab_traj, setup_traj), axis=0)
    # dim=2
    # ax.plot(tt[:t_left_1], grab_traj[:, dim], c='blue', linestyle='--')
    # ax.plot(tt[t_left_1:t_left_2], setup_traj[:, dim], c='red', linestyle='--')
    # ax.plot(tt[t_left_2:t_left_3], arc_traj_vs[:, dim], c='blue', linestyle='--')
    # ax.plot(tt[t_left_3:], arc_traj_vs2[:, dim], c='red', linestyle='--')
    # plt.show()

    # fig, ax = plt.subplots()
    # dim = 1
    # # ax.plot(tt[:t_left_1], grab_traj[:, dim], c='blue', linestyle='--')
    # # ax.plot(tt[t_left_1:t_left_2], setup_traj[:, dim], c='red', linestyle='--')
    # # ax.plot(tt[t_left_2:t_left_3], arc_traj_vs[:, dim], c='blue', linestyle='--')
    # # ax.plot(tt[t_left_2:t_left_3], xs, c='cyan', linestyle='-.')
    # ax.plot(arc_traj_vs[:, 1], arc_traj_vs[:, 2], c='blue', linestyle='--')
    # ax.plot(xs, arc_traj_vs[:, 2], c='cyan', linestyle='-.')
    # # ax.plot(tt[t_left_3:], arc_traj_vs2[:, dim], c='red', linestyle='--')
    # plt.show()

    # Ball trajectory
    # arc_traj_vs = arc_traj(data.site('shoulder1_left').xpos, r,
                            # 0, .9*np.pi/2, Tk_left_4, density_fn='')
    # arc_traj_ball = arc_traj(data.site('shoulder1_left').xpos, r, 0,
                             # 1.1*np.pi/2, Tk_left_4, density_fn='')

    # ball_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs), axis=0)
    ball_traj = left_arm_traj.copy()

    # ax.plot(tt[:t_left_1], grab_traj[:, 2], c='blue', linestyle='-')
    # ax.plot(tt[t_left_1:t_left_2], setup_traj[:, 2], c='red', linestyle='-')
    # ax.plot(tt[t_left_2:t_left_3], arc_traj_vs[:, 2], c='blue', linestyle='-')
    # ax.plot(tt[t_left_3:], arc_traj_vs2[:, 2], c='red', linestyle='--')
    # plt.show()

    time_dict = dict(
        Tk_right_1=Tk_right_1,
        Tk_right_2=Tk_right_2,
        Tk_right_3=Tk_right_3,
        Tk_left_1=Tk_left_1,
        Tk_left_2=Tk_left_2,
        Tk_left_3=Tk_left_3,
        t_right_1=t_right_1,
        t_left_1=t_left_1,
    )

    return right_arm_traj, left_arm_traj, ball_traj, time_dict

def tennis_traj(model, data, Tk):
    shouldxr = data.site('shoulder1_right').xpos
    shouldxl = data.site('shoulder1_left').xpos
    elbowx = data.site('elbow_right').xpos
    handxr = data.site('hand_right').xpos
    handxl = data.site('hand_left').xpos
    r1 = np.sum((shouldxr - elbowx)**2)**.5
    r2 = np.sum((elbowx - handxr)**2)**.5
    r = r1 + r2
    Tk_right_1 = int(Tk / 4) # Time to grab with right hand (1)
    Tk_right_2 = int(Tk / 12) # Time to grab with right hand (2)
    t_right_1 = Tk_right_1 + Tk_right_2
    Tk_right_3 = int(Tk / 4) # Time to set up
    t_right_2 = t_right_1 + Tk_right_3;
    Tk_right_4 = Tk - t_right_2 # Time to swing

    Tk_left_1 = int(Tk / 3) # Duration to grab with left hand (1)
    Tk_left_2 = int(Tk / 8) # Duration to grab with left hand (2)
    t_left_1 = Tk_left_1 + Tk_left_2 # Time up to end of grab
    Tk_left_3 = int(Tk / 6) # Duration to set up
    t_left_2 = t_left_1 + Tk_left_3 # Time to end of setting up
    Tk_left_4 = int(Tk / 10) # Duration to throw ball up
    t_left_3 = t_left_2 + Tk_left_4 # Time to end of throwing ball up
    Tk_left_5 = Tk - t_left_3 # Time to move hand down

    # Tk4 = int((Tk+Tk2)/2)

    # fig, ax = plt.subplots()
    # tt = np.linspace(0, 1, Tk)

    # Right arm

    # grab_targ = data.site('racket_handle').xpos + np.array([0, 0, -0.05])
    grab_targ = data.site('racket_handle_top').xpos + np.array([0, 0, 0.03])
    # grab_targ = data.site('racket_handle_top').xpos + np.array([0, 0, 0])
    sx = np.linspace(0, 1, Tk_right_1)
    s = sigmoid(sx, 5)
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk_right_2, 3))), axis=0)
    grab_traj = handxr + s*(grab_targ - handxr)

    arc_traj_vs = arc_traj(data.site('shoulder1_right').xpos, r, np.pi,
                                  np.pi/6, Tk_right_4, density_fn='')

    s = np.linspace(0, 1, Tk_right_3)
    s = sigmoid(s, 5)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])

    right_arm_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs),
                                    axis=0)

    t_fin = Tk * model.opt.timestep
    tt = np.linspace(0, t_fin, Tk)

    # fig, ax = plt.subplots()
    # ax.plot(tt[:t_right_1], grab_traj[:, 2], c='blue')
    # ax.plot(tt[t_right_1:t_right_2], setup_traj[:, 2], c='red')
    # ax.plot(tt[t_right_2:], arc_traj_vs[:, 2], c='blue')
    # plt.show()

    # Tk4 = int((Tk+Tk2)/2)

    # Left arm
    # grab_targ = data.site('ball').xpos + np.array([0, 0, 0.04])
    grab_targ = data.site('ball_top').xpos + np.array([0, 0, .03])
    s = sigmoid(np.linspace(0, 1, Tk_left_1), 5)
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk_left_2, 3))), axis=0)
    grab_traj = handxl + s*(grab_targ - handxl)

    arc_traj_vs = arc_traj(data.site('shoulder1_left').xpos, r,
                            -np.pi/8, .9*np.pi/2, Tk_left_4, density_fn='')
    xs = arc_traj_vs[:, 1].copy()
    x0 = xs[0]
    recenter_scale_xs = .8*(xs - x0)
    arc_traj_vs[:,1] = recenter_scale_xs + x0
    # arc_traj_vs2 = arc_traj(data.site('shoulder1_left').xpos, r,
                            # .9*np.pi/2, .7*np.pi/2, Tk_left_5, density_fn='')
    # arc_traj_vs2 = arc_traj_vs[:-Tk_left_5:-1]
    arc_traj_vs2 = arc_traj(data.site('shoulder1_left').xpos, r,
                            .9*np.pi/2, .7*np.pi/2, Tk_left_5, density_fn='')

    setup_traj = np.zeros((Tk_left_3, 3))
    s = np.linspace(0, 1, Tk_left_3)
    s = sigmoid(s, 5)
    # s = 2*sigmoid(.5*s, 5)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s*(arc_traj_vs[0] - grab_traj[-1])

    left_arm_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs,
                                    arc_traj_vs2), axis=0)
    # dim=2
    # ax.plot(tt[:t_left_1], grab_traj[:, dim], c='blue', linestyle='--')
    # ax.plot(tt[t_left_1:t_left_2], setup_traj[:, dim], c='red', linestyle='--')
    # ax.plot(tt[t_left_2:t_left_3], arc_traj_vs[:, dim], c='blue', linestyle='--')
    # ax.plot(tt[t_left_3:], arc_traj_vs2[:, dim], c='red', linestyle='--')
    # plt.show()

    # fig, ax = plt.subplots()
    # dim = 1
    # # ax.plot(tt[:t_left_1], grab_traj[:, dim], c='blue', linestyle='--')
    # # ax.plot(tt[t_left_1:t_left_2], setup_traj[:, dim], c='red', linestyle='--')
    # # ax.plot(tt[t_left_2:t_left_3], arc_traj_vs[:, dim], c='blue', linestyle='--')
    # # ax.plot(tt[t_left_2:t_left_3], xs, c='cyan', linestyle='-.')
    # ax.plot(arc_traj_vs[:, 1], arc_traj_vs[:, 2], c='blue', linestyle='--')
    # ax.plot(xs, arc_traj_vs[:, 2], c='cyan', linestyle='-.')
    # # ax.plot(tt[t_left_3:], arc_traj_vs2[:, dim], c='red', linestyle='--')
    # plt.show()

    # Ball trajectory
    # arc_traj_vs = arc_traj(data.site('shoulder1_left').xpos, r,
                            # 0, .9*np.pi/2, Tk_left_4, density_fn='')
    # arc_traj_ball = arc_traj(data.site('shoulder1_left').xpos, r, 0,
                             # 1.1*np.pi/2, Tk_left_4, density_fn='')

    # ball_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs), axis=0)
    ball_traj = left_arm_traj.copy()

    # ax.plot(tt[:t_left_1], grab_traj[:, 2], c='blue', linestyle='-')
    # ax.plot(tt[t_left_1:t_left_2], setup_traj[:, 2], c='red', linestyle='-')
    # ax.plot(tt[t_left_2:t_left_3], arc_traj_vs[:, 2], c='blue', linestyle='-')
    # ax.plot(tt[t_left_3:], arc_traj_vs2[:, 2], c='red', linestyle='--')
    # plt.show()

    time_dict = dict(
        Tk_right_1=Tk_right_1,
        Tk_right_2=Tk_right_2,
        Tk_right_3=Tk_right_3,
        Tk_right_4=Tk_right_4,
        Tk_left_1=Tk_left_1,
        Tk_left_2=Tk_left_2,
        Tk_left_3=Tk_left_3,
        Tk_left_4=Tk_left_4,
        Tk_left_5=Tk_left_5,
        t_right_1=t_right_1,
        t_right_2=t_right_2,
        t_left_1=t_left_1,
        t_left_2=t_left_2,
        t_left_3=t_left_3,
    )

    return right_arm_traj, left_arm_traj, ball_traj, time_dict

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
    two_arm_idx['adh_left_hand'] = acts[f'adh_left_hand']
    two_arm_idx['adh_right_hand'] = acts[f'adh_right_hand']
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

def get_idx_sets(env, exp_name):
    model = env.model
    data = env.data
    acts = opt_utils.get_act_ids(model)
    contact_check_list = []
    adh_ids = []
    let_go_ids = []
    if exp_name == 'basic_movements_right':
        throw_idxs = one_arm_idxs(model)
        site_grad_idxs = [throw_idxs['arm_a_without_adh']]
        stabilize_jnt_idx = throw_idxs['not_arm_j']
        stabilize_act_idx = throw_idxs['not_arm_a']
    elif exp_name == 'basic_movements_left':
        throw_idxs = one_arm_idxs(model, 'left')
        site_grad_idxs = [throw_idxs['arm_a_without_adh']]
        stabilize_jnt_idx = throw_idxs['not_arm_j']
        stabilize_act_idx = throw_idxs['not_arm_a']
    elif exp_name == 'basic_movements_both':
        tennis_idxs = two_arm_idxs(model)
        site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                          tennis_idxs['left_arm_without_adh']]
        stabilize_jnt_idx = tennis_idxs['not_arm_j']
        stabilize_act_idx = tennis_idxs['not_arm_a']
    elif exp_name == 'throw_ball':
        throw_idxs = one_arm_idxs(model)
        site_grad_idxs = [throw_idxs['arm_a_without_adh']]
        stabilize_jnt_idx = throw_idxs['not_arm_j']
        stabilize_act_idx = throw_idxs['not_arm_a']
        contact_check_list = [['ball', 'hand_right1'], ['ball', 'hand_right2']]
        adh_ids = [acts['adh_right_hand'][0], acts['adh_right_hand'][0]]
        let_go_ids = [acts['adh_right_hand'][0]]
    elif exp_name == 'grab_ball':
        throw_idxs = one_arm_idxs(model)
        site_grad_idxs = [throw_idxs['arm_a_without_adh']]
        stabilize_jnt_idx = throw_idxs['not_arm_j']
        stabilize_act_idx = throw_idxs['not_arm_a']
        contact_check_list = [['ball', 'hand_right1'], ['ball', 'hand_right2']]
        adh_ids = [acts['adh_right_hand'][0], acts['adh_right_hand'][0]]
        let_go_ids = []
        let_go_times = []
    elif exp_name == 'tennis_serve':
        tennis_idxs = two_arm_idxs(model)
        site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                          tennis_idxs['left_arm_without_adh'],
                          tennis_idxs['right_arm_without_adh'],
                          tennis_idxs['left_arm_without_adh']]
        site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                          tennis_idxs['left_arm_without_adh'],
                          tennis_idxs['right_arm_without_adh']]
        stabilize_jnt_idx = tennis_idxs['not_arm_j']
        stabilize_act_idx = tennis_idxs['not_arm_a']
        contact_check_list = [['racket_handle', 'hand_right1'], ['racket_handle', 'hand_right2'],
                              ['ball', 'hand_left1'], ['ball', 'hand_left2']]
        acts = opt_utils.get_act_ids(model)
        adh_ids = [acts['adh_right_hand'][0], acts['adh_right_hand'][0],
                   acts['adh_left_hand'][0], acts['adh_left_hand'][0]]
        act_ids = ['adh_right_hand', 'adh_right_hand', 'adh_left_hand',
                     'adh_left_hand']
        let_go_ids = [acts['adh_left_hand'][0]]
    elif exp_name == 'tennis_grab':
        tennis_idxs = two_arm_idxs(model)
        site_grad_idxs = [tennis_idxs['right_arm_without_adh'],
                          tennis_idxs['left_arm_without_adh'],
                          tennis_idxs['right_arm_without_adh']]
        stabilize_jnt_idx = tennis_idxs['not_arm_j']
        stabilize_act_idx = tennis_idxs['not_arm_a']

        contact_check_list = [['racket_handle', 'hand_right1'],
                              ['racket_handle', 'hand_right2'],
                              ['ball', 'hand_left1'], ['ball', 'hand_left2']]
        adh_ids = [acts['adh_right_hand'][0], acts['adh_right_hand'][0],
                   acts['adh_left_hand'][0], acts['adh_left_hand'][0]]
        act_ids = ['adh_right_hand', 'adh_right_hand',
                   'adh_left_hand', 'adh_left_hand']
        let_go_ids = []
        let_go_times = []
        

    out_dict = dict(site_grad_idxs=site_grad_idxs, stabilize_jnt_idx=stabilize_jnt_idx,
                    stabilize_act_idx=stabilize_act_idx, contact_check_list=contact_check_list,
                    adh_ids=adh_ids, let_go_ids=let_go_ids)
    return out_dict

def get_times(env, exp_name, Tf):
    model = env.model
    data = env.data
    dt = model.opt.timestep
    Tk = int(Tf / dt)
    time_dict = None
    grab_tk = 0
    let_go_times = []
    if exp_name == 'basic_movements_right':
        pass
    elif exp_name == 'throw_ball':
        time_dict = throw_traj(model, data, Tk)[-1]
        grab_t = Tf / 2.2
        grab_tk = int(grab_t/dt)
        let_go_times = [Tk]
    elif exp_name == 'grab_ball':
        out = throw_grab_traj(model, data, Tk)
        time_dict = out[1]
        grab_time = int(time_dict['t_1'] * .9)
        grab_t = Tf / 2.2
        grab_tk = int(grab_t/dt)
    elif exp_name == 'tennis_serve':
        time_dict = tennis_traj(model, data, Tk)[-1]
        grab_t = Tf / 2.2
        grab_tk = int(grab_t/dt)
        let_go_times = [time_dict['t_left_3']]
    out_dict = dict(grab_phase_tk=grab_tk, let_go_times=let_go_times)
    return out_dict


def make_traj_sets(env, exp_name, Tk):
    model = env.model
    data = env.data
    smoothing_sigma = int(.1 / model.opt.timestep)
    arc_std = 0.2
    joints = opt_utils.get_joint_ids(model)
    acts = opt_utils.get_act_ids(model)
    if exp_name == 'basic_movements_right':
        rs, thetas, wrist_qs = basic_movements.random_arcs_right_arm(
            model, data, Tk, data.site('hand_right').xpos, smoothing_sigma,
            arc_std)
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:,1] = rs * np.cos(thetas)
        traj1_xs[:,2] = rs * np.sin(thetas)
        traj1_xs += data.site('shoulder1_right').xpos
        full_traj = traj1_xs
        targ_traj_mask = np.ones((Tk,))
        targ_traj_mask_type = 'double_sided_progressive'

        sites = ['hand_right']
        target_trajs = [full_traj]
        masks = [targ_traj_mask]
        mask_types = [targ_traj_mask_type]

        q_targs = [np.zeros((Tk, model.nq))]
        q_targ_masks = [np.zeros((Tk, model.nq))]
        q_targ_mask_types = ['const']
    elif exp_name == 'basic_movements_left':
        rs, thetas, wrist_qs = basic_movements.random_arcs_left_arm(
            model, data, Tk, data.site('hand_left').xpos, smoothing_sigma,
            arc_std)
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:,1] = rs * np.cos(thetas)
        traj1_xs[:,2] = rs * np.sin(thetas)
        traj1_xs += data.site('shoulder1_left').xpos
        full_traj = traj1_xs
        targ_traj_mask = np.ones((Tk,))
        targ_traj_mask_type = 'double_sided_progressive'

        sites = ['hand_left']
        target_trajs = [full_traj]
        masks = [targ_traj_mask]
        mask_types = [targ_traj_mask_type]

        q_targs = [np.zeros((Tk, model.nq))]
        q_targ_masks = [np.zeros((Tk, model.nq))]
        q_targ_mask_types = ['const']
    elif exp_name == 'basic_movements_both':
        rs, thetas, wrist_qs = basic_movements.random_arcs_right_arm(
            model, data, Tk, data.site('hand_right').xpos, smoothing_sigma,
            arc_std)
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:,1] = rs * np.cos(thetas)
        traj1_xs[:,2] = rs * np.sin(thetas)
        traj1_xs += data.site('shoulder1_right').xpos
        full_traj = traj1_xs
        targ_traj_mask = np.ones((Tk,))
        targ_traj_mask_type = 'double_sided_progressive'

        target_trajs = [full_traj]
        masks = [targ_traj_mask]
        mask_types = [targ_traj_mask_type]

        rs, thetas, wrist_qs = basic_movements.random_arcs_left_arm(
            model, data, Tk, data.site('hand_left').xpos, smoothing_sigma,
            arc_std)
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:,1] = rs * np.cos(thetas)
        traj1_xs[:,2] = rs * np.sin(thetas)
        traj1_xs += data.site('shoulder1_left').xpos
        full_traj = traj1_xs
        targ_traj_mask = np.ones((Tk,))
        targ_traj_mask_type = 'double_sided_progressive'

        sites = ['hand_right', 'hand_left']
        target_trajs += [full_traj]
        masks += [targ_traj_mask]
        mask_types = ['double_sided_progressive', 'double_sided_progressive']

        q_targs = [np.zeros((Tk, model.nq)), np.zeros((Tk, model.nq))]
        q_targ_masks = [np.zeros((Tk, model.nq)), np.zeros((Tk, model.nq))]
        q_targ_mask_types = ['const', 'const']
    elif exp_name == 'throw_ball':
        targ_traj_mask = np.ones((Tk,))
        targ_traj_mask_type = 'double_sided_progressive'
        out = throw_traj(model, data, Tk)
        full_traj, time_dict = out


        bodyj = joints['body']['body_dofs']

        sites = ['hand_right']
        target_trajs = [full_traj]
        masks = [targ_traj_mask]
        mask_types = [targ_traj_mask_type]

        q_targ = np.zeros((Tk, 2*model.nq))
        q_targ_mask = np.zeros((Tk,2*model.nq))
        q_targ_mask2 = np.zeros((Tk,2*model.nq))
        q_targ_mask2[time_dict['t_1']:,
                    joints['all']['wrist_left']] = 1
        q_targ_nz = np.linspace(0, -2.44, time_dict['t_2']-time_dict['t_1'])
        q_targ[time_dict['t_1']:time_dict['t_2'], 
                joints['all']['wrist_left']] = q_targ_nz
        q_targ[time_dict['t_2']:, joints['all']['wrist_left']] = -2.44
        q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask, q_targ_mask]
        q_targ_mask_types = ['const']
        q_targs = [q_targ]
    elif exp_name == "grab_ball":
        targ_traj_mask = np.ones((Tk,))
        # targ_traj_mask_type = 'progressive'
        targ_traj_mask_type = 'double_sided_progressive'
        out = throw_grab_traj(model, data, Tk)
        full_traj, time_dict = out
        contact_check_list = [['ball', 'hand_right1'], ['ball', 'hand_right2']]
        adh_ids = [acts['adh_right_hand'][0], acts['adh_right_hand'][0]]
        let_go_ids = []
        # let_go_times = [Tk]
        let_go_times = []
        sites = ['hand_right']
        target_trajs = [full_traj]
        masks = [targ_traj_mask]
        mask_types = [targ_traj_mask_type]

        q_targ = np.zeros((Tk, 2*model.nq))
        q_targ_mask = np.zeros((Tk,2*model.nq))
        q_targ_mask2 = np.zeros((Tk,2*model.nq))
        q_targ_mask2[time_dict['t_1']:,
                    joints['all']['wrist_left']] = 1
        q_targ_nz = np.linspace(0, -2.44, time_dict['t_2']-time_dict['t_1'])
        q_targ[time_dict['t_1']:time_dict['t_2'], 
                joints['all']['wrist_left']] = q_targ_nz
        q_targ[time_dict['t_2']:, joints['all']['wrist_left']] = -2.44
        q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask, q_targ_mask]
        q_targ_mask_types = ['const']
        q_targs = [q_targ]
    elif exp_name == "tennis_serve":
        sites = ['hand_right', 'hand_left', 'racket_handle_top'] # Move
        targ_traj_mask = np.ones((Tk,))
        # targ_traj_mask_type = 'progressive'
        targ_traj_mask_type = 'double_sided_progressive'
        # targ_traj_mask_type = 'const'
        out = tennis_traj(model, data, Tk)
        right_hand_traj, left_hand_traj, ball_traj, time_dict = out
        ball_traj_mask = np.ones((Tk,))
        ball_traj_mask[time_dict['t_left_3']:] = 0
        out = tennis_traj(model, data, Tk)
        right_hand_traj, left_hand_traj, ball_traj, time_dict = out
        target_trajs = [right_hand_traj, left_hand_traj, right_hand_traj]
        masks = [targ_traj_mask, targ_traj_mask, targ_traj_mask]
        mask_types = [targ_traj_mask_type]*3
        q_targ = np.zeros((Tk, 2*model.nq))
        q_targ_mask = np.zeros((Tk,2*model.nq))
        q_targ_mask2 = np.zeros((Tk,2*model.nq))
        q_targ_mask2[time_dict['t_left_1']:, joints['all']['wrist_left']] = 1
        q_targ_nz = np.linspace(0, -2.44, time_dict['t_left_2']-time_dict['t_left_1'])
        q_targ[time_dict['t_left_1']:time_dict['t_left_2'], 
                joints['all']['wrist_left']] = q_targ_nz
        q_targ[time_dict['t_left_2']:, joints['all']['wrist_left']] = -2.44
        q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask, q_targ_mask]
        q_targ_mask_types = ['const']*3
        q_targs = [q_targ]*3
    elif exp_name == "tennis_grab":
        targ_traj_mask = np.ones((Tk,))
        targ_traj_mask_type = 'double_sided_progressive'
        out = tennis_grab_traj(model, data, Tk)
        right_hand_traj, left_hand_traj, ball_traj, time_dict = out
        sites = ['hand_right', 'hand_left', 'racket_handle_top']
        target_trajs = [right_hand_traj, left_hand_traj, right_hand_traj]
        masks = [targ_traj_mask, targ_traj_mask, targ_traj_mask]
        mask_types = [targ_traj_mask_type]*3
        q_targ = np.zeros((Tk, 2*model.nq))
        q_targ_mask = np.zeros((Tk,2*model.nq))
        q_targ_mask2 = np.zeros((Tk,2*model.nq))
        q_targ_mask2[time_dict['t_left_1']:,
                    joints['all']['wrist_left']] = 1
        q_targ_nz = np.linspace(0, -2.44, Tk-time_dict['t_left_1'])
        q_targ[time_dict['t_left_1']:, 
                joints['all']['wrist_left']] = q_targ_nz
        q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask, q_targ_mask]
        q_targ_mask_types = ['const']*3
        q_targs = [q_targ]*3


    out_dict = dict(sites=sites, target_trajs=target_trajs,
                    targ_traj_masks=masks,
                    targ_traj_mask_types=mask_types, q_targs=q_targs,
                    q_targ_masks=q_targ_masks,
                    q_targ_mask_types=q_targ_mask_types,
                    # time_dict=time_dict)
                   )

    return out_dict

def show_forward_sim(env, ctrls):
    for k in range(ctrls.shape[0]-1):
        util.step(env.model, env.data, ctrls[k])
        env.render()

def forward_with_sites(env, ctrls, site_names, render=False):
    n = len(site_names)
    site_xvs = np.zeros((n, ctrls.shape[0]+1, 3))
    model_qs = np.zeros((ctrls.shape[0]+1, env.model.nq))
    model_qs[0] = env.data.qpos.copy()
    for k2 in range(n):
        site_xvs[k2, 0] = env.data.site(site_names[k2]).xpos
    for k in range(ctrls.shape[0]):
        util.step(env.model, env.data, ctrls[k])
        model_qs[k+1] = env.data.qpos.copy()
        for k2 in range(n):
            site_xvs[k2, k+1] = env.data.site(site_names[k2]).xpos
        if render:
            env.render()
    return site_xvs, model_qs

def forward_to_contact(env, ctrls, noisev=None, render=True, let_go_times=[],
                       let_go_ids=[], n_steps_adh=10,
                       contact_check_list=[], adh_ids=[],
                      ):
    model = env.model
    act = opt_utils.get_act_ids(model)
    data = env.data
    ball_contact = False
    Tk = ctrls.shape[0]
    contact_cnt = 0
    contact = False
    adh_ctrl = opt_utils.AdhCtrl(let_go_times, let_go_ids, n_steps_adh,
                                 contact_check_list, adh_ids)
    if noisev is None:
        noisev = np.zeros((Tk, model.nu))
    contacts = np.zeros((Tk, 2))
    for k in range(Tk):
        ctrls[k], cont_k1, cont_k2 = adh_ctrl.get_ctrl(model, data, ctrls[k])
        contacts[k] = [cont_k1, cont_k2] # TODO: address this
        util.step(model, data, ctrls[k] + noisev[k])
        if render:
            env.render()
        # contact_pairs = util.get_contact_pairs(model, data)
        # for cp in contact_pairs:
            # if 'racket_handle' in cp and 'hand_right1' in cp or 'hand_right2' in cp:
                # contact = True
                # if contact_cnt <= 20:
                    # ctrls[k:, act['adh_right_hand']] = .05 * contact_cnt
                    # contact_cnt += 1
    return k, ctrls, contacts

class LimLowestDict:
    def __init__(self, max_len):
        self.max_len = max_len
        self.dict = sc.SortedDict()

    def append(self, key, val):
        self.dict.update({key: val})
        if len(self.dict) > self.max_len:
            self.dict.popitem()

class DoubleSidedProgressive:
    def __init__(self, incr_every, amnt_to_incr, grab_phase_it, grab_phase_tk,
                 phase_2_it, max_idx=1e8):
        self.incr_every = incr_every
        self.amnt_to_incr = amnt_to_incr
        self.k = 0
        self.incr_k = 0
        self.incr_cnt = 0
        self.incr_cnt2 = 0
        self.phase_2_it = phase_2_it
        self.grab_phase_it = grab_phase_it
        self.grab_phase_tk = grab_phase_tk
        self.grab_end_idx = 0
        self.phase = 'grab'

    def _update_grab_phase(self):
        start_idx = 0
        end_idx = self.grab_phase_tk
        self.grab_end_idx = end_idx
        return slice(start_idx, end_idx)
    
    def _update_phase_1(self):
        start_idx = self.grab_end_idx
        end_idx = self.amnt_to_incr*(self.incr_cnt+1) + start_idx
        idx = slice(start_idx, end_idx)
        self.incr_cnt += 1
        return idx

    def _update_phase_2(self):
        start_idx = self.amnt_to_incr*self.incr_cnt2 + self.grab_end_idx
        end_idx = self.amnt_to_incr*(self.incr_cnt+1) + start_idx
        self.incr_cnt2 += 1
        idx = slice(start_idx, end_idx)
        return idx

    def update(self):
        if self.k < self.grab_phase_it:
            self.idx = self._update_grab_phase()
        elif self.k == self.grab_phase_it:
            self.phase = 'phase_1'
            self.incr_k = 0
        if self.phase != 'grab' and self.incr_k % self.incr_every == 0:
            if self.k >= self.phase_2_it:
                self.phase = 'phase_2'
                self.idx = self._update_phase_2()
            else:
                self.phase = 'phase_1'
                self.idx = self._update_phase_1()
        self.k += 1
        self.incr_k += 1
        return self.idx

class WindowedIdx:
    def __init__(self, incr_every, amnt_to_incr, window_size, max_idx=1e8):
        self.incr_every = incr_every
        self.amnt_to_incr = amnt_to_incr
        self.k = 0
        self.incr_cnt = 0
        self.incr_cnt2 = 0
        self.window_size = window_size
        self.idx = None
    
    def update(self):
        end_idx = self.amnt_to_incr*(self.incr_cnt+1)
        start_idx = max(0, end_idx - self.window_size)
        if self.k % self.incr_every == 0:
            self.idx = slice(start_idx, end_idx)
            self.incr_cnt += 1
        self.k += 1
        return self.idx

def show_plot(hxs, target_trajs, targ_traj_mask_currs,
              # qs, qs_targs,
              site_names,
              site_grad_idxs, ctrls, axs, grads, tt, show=True, save=False):
    fig = axs[0, 0].figure
    n = len(site_names)
    nr = range(n)
    for k in nr:
        hx = hxs[k]
        tm = np.tile(targ_traj_mask_currs[k], (3, 1)).T
        tm[tm == 0] = np.nan
        ft = target_trajs[k]*tm
        loss = np.mean((hx - target_trajs[k])**2)
        ax = axs[0, k]
        ax.cla()
        ax.plot(tt, hx[:,1], color='blue', label='x')
        ax.plot(tt, ft[:,1], '--', color='blue')
        ax.plot(tt, hx[:,2], color='red', label='y')
        ax.plot(tt, ft[:,2], '--', color='red')
        # if k == 1:
            # ax.plot(tt, qs, color='green', label='qs')
            # ax.plot(tt, qs_targs, '--',
                    # color='green', label='qs_targ')
        # ax.set_title(site_names[k] + ' loss: ' + str(loss))
        ax.set_title(site_names[k])
        ax.legend()
        ax = axs[1,k]
        ax.cla()
        ax.plot(tt[:-1], ctrls[:, site_grad_idxs[k]])
        ax = axs[2,k]
        ax.cla()
        ax.plot(tt[:-1], grads[k])
    axs[1,0].plot(tt[:-1], ctrls[:, -2])
    # axs[1,1].plot(tt[:-1], ctrls[:, -1])
    fig.tight_layout()
    if show:
        plt.show(block=False)
        plt.pause(.05)
    if save:
        fig.savefig('fig.pdf')

def arm_target_traj(env, sites, site_grad_idxs, stabilize_jnt_idx,
                    stabilize_act_idx, target_trajs, targ_traj_masks,
                    targ_traj_mask_types, q_targs, q_targ_masks,
                    q_targ_mask_types, ctrls, grad_trunc_tk, seed,
                    ctrl_rate, ctrl_std, Tk, max_its=30, lr=10, lr2=10,
                    it_lr2=31, keep_top=1,
                    incr_every=5, amnt_to_incr=5, grad_update_every=1,
                    grab_phase_it=0, grab_phase_tk=0,
                    phase_2_it=None,
                    update_plot_every=1, optimizer='adam',
                    contact_check_list=[], adh_ids=[],
                    balance_cost=1000, joint_cost=100,
                    let_go_times=[],
                    let_go_ids=[],
                    n_steps_adh=10,
                   ):
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
    if phase_2_it is None:
        phase_2_it = max_its+1
    model = env.model
    data = env.data


    not_stabilize_act_idx = [k for k in range(model.nu) if k not in
                             stabilize_act_idx]

    n_sites = len(sites)
    assert (n_sites == len(target_trajs) and n_sites == len(targ_traj_masks)
            and n_sites == len(targ_traj_mask_types))

    data0 = copy.deepcopy(data)


    noisev = make_noisev(model, seed, Tk, ctrl_std, ctrl_rate)

    qs, qvels, ss = util.forward_sim(model, data, ctrls + noisev)
    util.reset_state(model, data, data0)

    ### Gradient descent
    qpos0 = data.qpos.copy()

    dt = model.opt.timestep
    T = Tk*dt
    tt = np.arange(0, T, dt)

    joints = opt_utils.get_joint_ids(model)
    # q_targs_wr = q_targs[1][:, joints['all']['wrist_left']]

    progbar = util.ProgressBar(final_it = max_its) 

    def get_opt(lr):
        if optimizer == 'rmsprop':
            return opts.RMSProp(lr=lr)
        if optimizer == 'adam':
            return opts.Adam(lr=lr)
        if optimizer == 'mom_sgd':
            return opts.SGD(lr=lr, momentum=0.2)
        if optimizer == 'sgd':
            return opts.SGD(lr=lr, momentum=0.2)

    optms = []
    targ_traj_progs = []
    targ_traj_mask_currs = []
    incr_cnts = []
    incr_everys = []
    amnt_to_incrs = []
    idxs = [0]*n_sites
    # targ_traj_masks_grab = np.zeros((Tk, n_sites))
    targ_traj_masks_grab = [0]*n_sites
    # targ_traj_masks_grab[:grab_time, :] = targ_traj_masks[:
    for k in range(n_sites):
        optms.append(get_opt(lr))
        if targ_traj_mask_types[k] == 'double_sided_progressive':
            idxs[k] = DoubleSidedProgressive(incr_every, amnt_to_incr,
                                             grab_phase_it, grab_phase_tk,
                                             phase_2_it=phase_2_it)
            # idxs[k] = WindowedIdx(incr_every, amnt_to_incr, 10*amnt_to_incr)
        targ_traj_progs.append((isinstance(targ_traj_mask_types[k], str)
                                  and targ_traj_mask_types[k] == 'progressive'))
        targ_traj_mask_currs.append(targ_traj_masks[k])
        if targ_traj_progs[k]:
            targ_traj_mask_currs[k] = np.zeros((Tk,))
            incr_cnts.append(0)
    lowest_losses = LimLowestDict(keep_top)
    lowest_losses_curr_mask = LimLowestDict(keep_top)

    # plt.ion()
    # fig, axs = plt.subplots(1, n_sites, figsize=(5*n_sites, 5))
    contact_bool = False
    # fig, axs = plt.subplots(2, n_sites, figsize=(5*n_sites, 5))
    grab_phase_switch = True
    fig, axs = plt.subplots(3, n_sites, figsize=(5*n_sites, 5))
    if n_sites == 1:
        axs = axs.reshape((3,1))
    try:
        for k0 in range(max_its):
            if k0 >= it_lr2:
                lr = lr2
            if k0 % incr_every == 0:
                for k in range(n_sites):
                    optms[k] = get_opt(lr)
            progbar.update(' ' + str(k0))
            for k in range(n_sites):
                # if False:
                # if not contact_bool and k0 > 100:
                targ_traj_mask_currs[k] = np.zeros((Tk,))
                idx = idxs[k].update()
                targ_traj_mask_currs[k][idx] = targ_traj_masks[k][idx]
            # if (idxs[0].incr_k-1) % incr_every == 0:
                # for k in range(n_sites):
                    # optms[k] = get_opt(lr)
            if idxs[0].phase != 'grab' and grab_phase_switch:
                grab_phase_switch = False
                print("End of grab phase. Selecting best ctrls.")
                if len(lowest_losses_curr_mask.dict) > 0:
                    ctrls = lowest_losses_curr_mask.dict.peekitem(0)[1][1]
                # util.reset_state(model, data, data0)
                # hxs = forward_with_sites(env, ctrls, site_names, False)
                # show_plot(hxs, target_trajs, targ_traj_mask_currs, site_names,
                          # site_grad_idxs, ctrls, axs, grads, tt)
                # util.reset_state(model, data, data0)
            # if k0 % incr_every == 0:
                
                # if targ_traj_progs[k] and k0 % incr_every == 0:
                    # idx = slice(amnt_to_incr*incr_cnts[k],
                                # amnt_to_incr*(incr_cnts[k]+1))
                    # targ_traj_mask_currs[k][idx] = targ_traj_masks[k][idx]
                    # incr_cnts[k] += 1

            util.reset_state(model, data, data0)
            k, ctrls, contacts = forward_to_contact(
                env, ctrls, noisev, False, let_go_times, let_go_ids,
                n_steps_adh, contact_check_list, adh_ids)
            contact_bool = np.sum(contacts[:, 0]) * np.sum(contacts[:, 1]) > 0
            # if ball_contact:
            util.reset_state(model, data, data0)
            grads = [0] * n_sites
            hxs = [0] * n_sites
            dldss = [0] * n_sites
            losses = [0] * n_sites
            losses_curr_mask = [0] * n_sites
            update_phase = k0 % grad_update_every
            tic = time.time()
            # if k0 == 160:
            # grads = opt_utils.traj_deriv_new2(
                # model, data, ctrls + noisev, target_trajs,
                # targ_traj_mask_currs,
                # q_targs, q_targ_masks,
                # grad_trunc_tk,
                # deriv_sites=site_names,
                # deriv_id_lists=site_grad_idxs,
                # update_every=grad_update_every, update_phase=update_phase,
                # grab_time=grab_time,
                # let_go_time=let_go_time
            # )
            # util.reset_state(model, data, data0)
            for k in range(n_sites):
                grads[k] = opt_utils.traj_deriv_new(
                    model, data, ctrls + noisev, target_trajs[k],
                    targ_traj_mask_currs[k],
                    q_targs[k], q_targ_masks[k],
                    grad_trunc_tk,
                    deriv_ids=site_grad_idxs[k], deriv_site=sites[k],
                    update_every=grad_update_every, update_phase=update_phase,
                    let_go_times=let_go_times,
                    let_go_ids=let_go_ids,
                    n_steps_adh=n_steps_adh,
                    contact_check_list=contact_check_list,
                    adh_ids=adh_ids
                )
                util.reset_state(model, data, data0)
            # grads[0][:, :grab_time] *= 4
            # if np.max(np.abs(grads)) > 5:
                # print('big_grad', np.max(np.abs(grads)))
                # for k in range(n_sites):
                    # ctrls[:, site_grad_idxs[k]] += 1e-6*np.random.randn(
                        # Tk-1, len(site_grad_idxs[k]))
            # else:
                # for k in range(n_sites):
                    # # ctrls[:, right_arm_without_adh] = optm.update(
                        # # ctrls[:, right_arm_without_adh], grads1[:Tk-1], 'ctrls', loss1)
                    # ctrls[:, site_grad_idxs[k]] = optms[k].update(
                        # ctrls[:, site_grad_idxs[k]], grads[k][:Tk-1], 'ctrls',
                        # losses[k])
            # if np.max(np.abs(grads)) > 5:
                # print(np.max(np.abs(grads)))
            for k in range(n_sites):
                ctrls[:, site_grad_idxs[k]] = optms[k].update(
                    ctrls[:, site_grad_idxs[k]], grads[k][:Tk-1], 'ctrls',
                        losses[k])

            ctrls = np.clip(ctrls, -1, 1)
            try:
                ctrls, __, qs, qvels = opt_utils.get_stabilized_ctrls(
                    model, data, Tk, noisev, qpos0, stabilize_act_idx,
                    stabilize_jnt_idx, ctrls[:, not_stabilize_act_idx],
                    K_update_interv=500, balance_cost=balance_cost, 
                    joint_cost=joint_cost, let_go_times=let_go_times,
                    let_go_ids = let_go_ids,
                    n_steps_adh=n_steps_adh,
                )
            except np.linalg.LinAlgError:
                print("LinAlgError in get_stabilized_ctrls")
                ctrls[:, not_stabilize_act_idx] *= .99
            ctrls = np.clip(ctrls, -1, 1)
            util.reset_state(model, data, data0)
            hxs, qs = forward_with_sites(env, ctrls, sites, True)
            for k in range(n_sites):
                hx = hxs[k]
                diffsq = (hx - target_trajs[k])**2
                losses[k] = np.mean(diffsq)
                mask = np.tile((targ_traj_mask_currs[k]>0), (3, 1)).T
                temp = np.sum(diffsq*mask) / (np.sum(mask[:,0]))
                losses_curr_mask[k] = temp
            # ctrls, __, qs, qvels = opt_utils.get_stabilized_ctrls(
                # model, data, Tk, noisev, qpos0, not_arm_a,
                # not_arm_j, ctrls[:, arm_a],
            # )
            loss = sum([loss.item() for loss in losses]) / n_sites
            lowest_losses.append(loss, (k0, ctrls.copy()))
            loss_curr_mask_avg = sum([loss.item() for loss in losses_curr_mask]) / n_sites 
            lowest_losses_curr_mask.append(loss_curr_mask_avg, (k0, ctrls.copy()))
            print(loss_curr_mask_avg)
            toc = time.time()
            # print(loss, toc-tic)
            nr = range(n_sites)
            if k0 % update_plot_every == 0 or k0 % incr_every == 0:
                # qs_wr = qs[:, joints['all']['wrist_left']]
                show_plot(hxs, target_trajs, targ_traj_mask_currs,
                          # qs_wr,
                          # q_targs_wr,
                          sites, site_grad_idxs, ctrls, axs,
                          grads, tt, show=False, save=True)
            # util.reset_state(model, data, data0)
            # k, ctrls = forward_to_contact(env, ctrls, noisev, True)
                    # plt.show()
                    # if k0 > phase_2_it:

            # util.reset_state(model, data, data0)
            # hx = forward_with_site(env, ctrls, site_names[0], True)
    finally:
        pass
    # except KeyboardInterrupt:
        # pass

    return ctrls, lowest_losses.dict
