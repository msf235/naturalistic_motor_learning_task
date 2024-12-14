from typing import Dict, List, Any
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
import masks

# Site names
RHAND_S = "R_Hand"
LHAND_S = "L_Hand"
RFOOT_S = "R_Ankle"
LFOOT_S = "L_Ankle"
RSHOULD_S = "R_Shoulder"
LSHOULD_S = "L_Shoulder"
RELBOW_S = "R_Elbow"
LELBOW_S = "L_Elbow"


def make_noisev(model, seed, Tk, CTRL_STD, CTRL_RATE):
    acts = opt_utils.get_act_ids(model)
    adh = acts["adh_right_hand"]
    rng = np.random.default_rng(seed)
    width = int(CTRL_RATE / model.opt.timestep)
    kernel = np.exp(-0.5 * np.linspace(-3, 3, width) ** 2)
    kernel /= np.linalg.norm(kernel)
    noise = util.FilteredNoise(model.nu, kernel, rng)
    noisev = CTRL_STD * noise.sample(Tk - 1)
    noisev[:, adh] = 0
    return noisev


def arc_traj(x0, r, theta0, theta1, n, density_fn="uniform"):
    if density_fn != "uniform":
        unif = np.linspace(0, 1, n)
        theta = (theta1 - theta0) * unif**1.5 + theta0
    else:
        theta = np.linspace(theta0, theta1, n)

    x = x0 + r * np.array([0 * theta, np.cos(theta), np.sin(theta)]).T
    return x


def sigmoid(x, a):
    # return .5 * (np.tanh(x-.5) + 1)
    return 0.5 * np.tanh(a * (x - 0.5)) + 0.5


def throw_grab_traj(model, data, Tk):
    shouldx = data.site(RSHOULD_S).xpos
    elbowx = data.site(RELBOW_S).xpos
    handx = data.site(RHAND_S).xpos
    r1 = np.sum((shouldx - elbowx) ** 2) ** 0.5
    r2 = np.sum((elbowx - handx) ** 2) ** 0.5
    r = r1 + r2
    Tk1 = int(Tk / 3)
    # Tk2 = int(2*Tk/3)
    Tk2 = Tk - Tk1
    Tk3 = int((Tk + Tk2) / 2)
    arc_traj_vs = arc_traj(
        data.site(RSHOULD_S).xpos, r, np.pi, np.pi / 2.2, Tk - Tk2, density_fn=""
    )
    grab_targ = data.site("ball").xpos + np.array([0, 0, 0])
    s = sigmoid(np.linspace(0, 1, Tk1), 5)
    s = np.tile(s, (3, 1)).T
    grab_traj = handx + s * (grab_targ - handx)
    # grab_traj[-1] = grab_targ

    setup_traj = np.zeros((Tk2, 3))
    s = np.linspace(0, 1, Tk2)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s * (arc_traj_vs[0] - grab_traj[-1])
    full_traj = np.concatenate((grab_traj, setup_traj), axis=0)

    time_dict = {
        "t_1": Tk1,
        "t_2": Tk2,
        "t_3": Tk3,
        "Tk1": Tk1,
        "Tk2": Tk2 - Tk1,
        "Tk3": Tk3 - Tk2,
    }

    return full_traj, time_dict


def throw_traj(model, data, Tk):
    shouldx = data.site(RSHOULD_S).xpos
    elbowx = data.site(RELBOW_S).xpos
    handx = data.site(RHAND_S).xpos
    r1 = np.sum((shouldx - elbowx) ** 2) ** 0.5
    r2 = np.sum((elbowx - handx) ** 2) ** 0.5
    r = r1 + r2
    Tk1 = int(Tk / 3)
    Tk2 = int(2 * Tk / 3)
    Tk3 = int((Tk + Tk2) / 2)
    arc_traj_vs = arc_traj(
        data.site(RSHOULD_S).xpos, r, np.pi, np.pi / 2.2, Tk - Tk2, density_fn=""
    )
    grab_targ = data.site("ball").xpos + np.array([0, 0, 0])
    s = sigmoid(np.linspace(0, 1, Tk1), 5)
    s = np.tile(s, (3, 1)).T
    grab_traj = handx + s * (grab_targ - handx)
    # grab_traj[-1] = grab_targ

    setup_traj = np.zeros((Tk2, 3))
    s = np.linspace(0, 1, Tk2 - Tk1)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s * (arc_traj_vs[0] - grab_traj[-1])
    full_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs), axis=0)

    time_dict = {
        "t_1": Tk1,
        "t_2": Tk2,
        "t_3": Tk3,
        "Tk1": Tk1,
        "Tk2": Tk2 - Tk1,
        "Tk3": Tk3 - Tk2,
    }

    return full_traj, time_dict


def tennis_grab_traj(model, data, Tk):
    shouldxr = data.site(RSHOULD_S).xpos
    shouldxl = data.site(LSHOULD_S).xpos
    elbowx = data.site(RELBOW_S).xpos
    handxr = data.site(RHAND_S).xpos
    handxl = data.site(LHAND_S).xpos
    r1 = np.sum((shouldxr - elbowx) ** 2) ** 0.5
    r2 = np.sum((elbowx - handxr) ** 2) ** 0.5
    r = r1 + r2
    Tk_right_1 = int(Tk / 4)  # Time to grab with right hand (1)
    Tk_right_2 = int(Tk / 12)  # Time to grab with right hand (2)
    t_right_1 = Tk_right_1 + Tk_right_2
    Tk_right_3 = Tk - t_right_1

    Tk_left_1 = int(Tk / 3)  # Duration to grab with left hand (1)
    Tk_left_2 = int(Tk / 8)  # Duration to grab with left hand (2)
    t_left_1 = Tk_left_1 + Tk_left_2  # Time up to end of grab
    Tk_left_3 = Tk - t_left_1  # Duration to set up

    # Tk4 = int((Tk+Tk2)/2)

    # fig, ax = plt.subplots()
    # tt = np.linspace(0, 1, Tk)

    # Right arm

    # grab_targ = data.site('racket_handle').xpos + np.array([0, 0, -0.05])
    grab_targ = data.site("racket_handle_top").xpos + np.array([0, 0, 0.03])
    # grab_targ = data.site('racket_handle_top').xpos + np.array([0, 0, 0])
    sx = np.linspace(0, 1, Tk_right_1)
    s = sigmoid(sx, 5)
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk_right_2, 3))), axis=0)
    grab_traj = handxr + s * (grab_targ - handxr)

    arc_traj_vs = arc_traj(
        data.site(RSHOULD_S).xpos, r, np.pi, np.pi / 6, 10, density_fn=""
    )

    s = np.linspace(0, 1, Tk_right_3)
    s = sigmoid(s, 5)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s * (arc_traj_vs[0] - grab_traj[-1])

    right_arm_traj = np.concatenate((grab_traj, setup_traj), axis=0)

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
    grab_targ = data.site("ball_top").xpos + np.array([0, 0, 0.03])
    s = sigmoid(np.linspace(0, 1, Tk_left_1), 5)
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk_left_2, 3))), axis=0)
    grab_traj = handxl + s * (grab_targ - handxl)

    arc_traj_vs = arc_traj(
        data.site(LSHOULD_S).xpos, r, np.pi / 5, np.pi / 2, 10, density_fn=""
    )
    xs = arc_traj_vs[:, 1].copy()
    x0 = xs[0]
    recenter_scale_xs = 0.8 * (xs - x0)
    arc_traj_vs[:, 1] = recenter_scale_xs + x0
    # arc_traj_vs2 = arc_traj(data.site(LSHOULD_S).xpos, r,
    # .9*np.pi/2, .7*np.pi/2, Tk_left_5, density_fn='')
    # arc_traj_vs2 = arc_traj_vs[:-Tk_left_5:-1]
    arc_traj_vs2 = arc_traj(
        data.site(LSHOULD_S).xpos,
        r,
        0.9 * np.pi / 2,
        0.7 * np.pi / 2,
        10,
        density_fn="",
    )

    setup_traj = np.zeros((Tk_left_3, 3))
    s = np.linspace(0, 1, Tk_left_3)
    s = sigmoid(s, 5)
    # s = 2*sigmoid(.5*s, 5)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s * (arc_traj_vs[0] - grab_traj[-1])

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
    # arc_traj_vs = arc_traj(data.site(LSHOULD_S).xpos, r,
    # 0, .9*np.pi/2, Tk_left_4, density_fn='')
    # arc_traj_ball = arc_traj(data.site(LSHOULD_S).xpos, r, 0,
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
    shouldxr = data.site(RSHOULD_S).xpos
    shouldxl = data.site(LSHOULD_S).xpos
    elbowx = data.site(RELBOW_S).xpos
    handxr = data.site(RHAND_S).xpos
    handxl = data.site(LHAND_S).xpos
    r1 = np.sum((shouldxr - elbowx) ** 2) ** 0.5
    r2 = np.sum((elbowx - handxr) ** 2) ** 0.5
    r = r1 + r2
    Tk_right_1 = int(Tk / 4)  # Time to grab with right hand (1)
    Tk_right_2 = int(Tk / 12)  # Time to grab with right hand (2)
    t_right_1 = Tk_right_1 + Tk_right_2
    Tk_right_3 = int(Tk / 4)  # Time to set up
    t_right_2 = t_right_1 + Tk_right_3
    Tk_right_4 = Tk - t_right_2  # Time to swing

    Tk_left_1 = int(Tk / 3)  # Duration to grab with left hand (1)
    Tk_left_2 = int(Tk / 8)  # Duration to grab with left hand (2)
    t_left_1 = Tk_left_1 + Tk_left_2  # Time up to end of grab
    Tk_left_3 = int(Tk / 6)  # Duration to set up
    t_left_2 = t_left_1 + Tk_left_3  # Time to end of setting up
    Tk_left_4 = int(Tk / 10)  # Duration to throw ball up
    t_left_3 = t_left_2 + Tk_left_4  # Time to end of throwing ball up
    Tk_left_5 = Tk - t_left_3  # Time to move hand down

    # Tk4 = int((Tk+Tk2)/2)

    # fig, ax = plt.subplots()
    # tt = np.linspace(0, 1, Tk)

    # Right arm

    # grab_targ = data.site('racket_handle').xpos + np.array([0, 0, -0.05])
    grab_targ = data.site("racket_handle_top").xpos + np.array([0, 0, 0.01])
    # grab_targ = data.site('racket_handle_top').xpos + np.array([0, 0, 0])
    sx = np.linspace(0, 1, Tk_right_1)
    s = sigmoid(sx, 5)
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk_right_2, 3))), axis=0)
    grab_traj = handxr + s * (grab_targ - handxr)

    arc_traj_vs = arc_traj(
        data.site(RSHOULD_S).xpos, r, np.pi, np.pi / 6, Tk_right_4, density_fn=""
    )

    s = np.linspace(0, 1, Tk_right_3)
    s = sigmoid(s, 5)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s * (arc_traj_vs[0] - grab_traj[-1])

    right_arm_traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs), axis=0)

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
    grab_targ = data.site("ball_top").xpos + np.array([0, 0, 0.01])
    s = sigmoid(np.linspace(0, 1, Tk_left_1), 5)
    s = np.tile(s, (3, 1)).T
    s = np.concatenate((s, np.ones((Tk_left_2, 3))), axis=0)
    grab_traj = handxl + s * (grab_targ - handxl)

    arc_traj_vs = arc_traj(
        data.site(LSHOULD_S).xpos,
        r,
        -np.pi / 8,
        0.9 * np.pi / 2,
        Tk_left_4,
        density_fn="",
    )
    xs = arc_traj_vs[:, 1].copy()
    x0 = xs[0]
    recenter_scale_xs = 0.8 * (xs - x0)
    arc_traj_vs[:, 1] = recenter_scale_xs + x0
    # arc_traj_vs2 = arc_traj(data.site(LSHOULD_S).xpos, r,
    # .9*np.pi/2, .7*np.pi/2, Tk_left_5, density_fn='')
    # arc_traj_vs2 = arc_traj_vs[:-Tk_left_5:-1]
    arc_traj_vs2 = arc_traj(
        data.site(LSHOULD_S).xpos,
        r,
        0.9 * np.pi / 2,
        0.7 * np.pi / 2,
        Tk_left_5,
        density_fn="",
    )

    setup_traj = np.zeros((Tk_left_3, 3))
    s = np.linspace(0, 1, Tk_left_3)
    s = sigmoid(s, 5)
    # s = 2*sigmoid(.5*s, 5)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s * (arc_traj_vs[0] - grab_traj[-1])

    left_arm_traj = np.concatenate(
        (grab_traj, setup_traj, arc_traj_vs, arc_traj_vs2), axis=0
    )
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
    # arc_traj_vs = arc_traj(data.site(LSHOULD_S).xpos, r,
    # 0, .9*np.pi/2, Tk_left_4, density_fn='')
    # arc_traj_ball = arc_traj(data.site(LSHOULD_S).xpos, r, 0,
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
    body_j = opt_utils.get_joint_ids(model)["body"]
    acts = opt_utils.get_act_ids(model)

    # body_j = joints['body']['dofadrs']
    # body_j = joints['body']
    # two_arm_idx['body_j'] = opt_utils.convert_dofadr(model, None,
    # joints['body_dofs'])
    raj = body_j["right_arm_dofadrs"]
    # raj = opt_utils.convert_dofadr(model, None, joints['right_arm'])
    laj = body_j["left_arm_dofadrs"]
    # laj = opt_utils.convert_dofadr(model, None, joints['left_arm'])
    arm_dofadrs = [k for k in body_j if k in raj or k in body_j["left_arm"]]
    two_arm_idx["not_arm_j"] = [i for i in body_j if i not in arm_dofadrs]
    arm_a = [k for k in acts["all"] if k in acts["right_arm"] or k in acts["left_arm"]]
    two_arm_idx["not_arm_a"] = [
        k for k in acts["all"] if k not in arm_a and k not in acts["adh"]
    ]
    two_arm_idx["right_arm_without_adh"] = [
        k for k in acts["right_arm"] if k not in acts["adh"]
    ]
    two_arm_idx["left_arm_without_adh"] = [
        k for k in acts["left_arm"] if k not in acts["adh"]
    ]
    two_arm_idx["adh_left_hand"] = acts[f"adh_left_hand"]
    two_arm_idx["adh_right_hand"] = acts[f"adh_right_hand"]
    return two_arm_idx


def one_arm_idxs(model, right_or_left="right"):
    joints = opt_utils.get_joint_ids(model)
    acts = opt_utils.get_act_ids(model)

    def ints(l1, l2):
        return list(set(l1).intersection(set(l2)))

    one_arm_idx = {}

    arm_dofadrs = joints["body"][f"{right_or_left}_arm_dofadrs"]
    # not_arm_dofadrs = [i for i in joints['body']['dofadrs'] if i not in arm_dofadrs]
    # one_arm_idx['not_arm_dofadrs'] = [i for i in joints['body']['dofadrs_without_root']
    # if i not in arm_dofadrs]
    one_arm_idx["not_arm_dofadrs"] = [
        i for i in joints["body"]["dofadrs"] if i not in arm_dofadrs
    ]
    arm_act = acts[f"{right_or_left}_arm"]
    arm_act_without_adh = [k for k in arm_act if k not in acts["adh"]]
    # Include all adhesion (including other hand)
    arm_with_all_adh = [k for k in acts["all"] if k in arm_act or k in acts["adh"]]
    not_arm_act = [k for k in acts["all"] if k not in arm_act and k not in acts["adh"]]
    one_arm_idx["arm_act_without_adh"] = arm_act_without_adh
    one_arm_idx["not_arm_act"] = not_arm_act
    return one_arm_idx


def get_idx_sets(env, exp_name):
    model = env.model
    data = env.data
    acts = opt_utils.get_act_ids(model)
    contact_check_list = []
    adh_ids = []
    let_go_ids = []
    if exp_name == "basic_movements_right":
        sites = [RHAND_S]
        throw_idxs = one_arm_idxs(model, "right")
        site_grad_idxs = [throw_idxs["arm_act_without_adh"]]
        stabilize_jnt_idx = throw_idxs["not_arm_dofadrs"]
        stabilize_act_idx = throw_idxs["not_arm_act"]
        other_act_idx = throw_idxs["arm_act_without_adh"]
    elif exp_name == "basic_movements_left":
        sites = [LHAND_S]
        throw_idxs = one_arm_idxs(model, "left")
        site_grad_idxs = [throw_idxs["arm_a_without_adh"]]
        stabilize_jnt_idx = throw_idxs["not_arm_j"]
        stabilize_act_idx = throw_idxs["not_arm_a"]
    elif exp_name == "basic_movements_both":
        sites = [RHAND_S, LHAND_S]
        tennis_idxs = two_arm_idxs(model)
        site_grad_idxs = [
            tennis_idxs["right_arm_without_adh"],
            tennis_idxs["left_arm_without_adh"],
        ]
        stabilize_jnt_idx = tennis_idxs["not_arm_j"]
        stabilize_act_idx = tennis_idxs["not_arm_a"]
    elif exp_name == "throw_ball":
        sites = [RHAND_S]
        throw_idxs = one_arm_idxs(model)
        site_grad_idxs = [throw_idxs["arm_a_without_adh"]]
        stabilize_jnt_idx = throw_idxs["not_arm_j"]
        stabilize_act_idx = throw_idxs["not_arm_a"]
        contact_check_list = [["ball", "hand_right1"], ["ball", "hand_right2"]]
        adh_ids = [acts["adh_right_hand"][0], acts["adh_right_hand"][0]]
        let_go_ids = [acts["adh_right_hand"][0]]
    elif exp_name == "grab_ball":
        sites = [RHAND_S]
        throw_idxs = one_arm_idxs(model)
        site_grad_idxs = [throw_idxs["arm_a_without_adh"]]
        stabilize_jnt_idx = throw_idxs["not_arm_j"]
        stabilize_act_idx = throw_idxs["not_arm_a"]
        contact_check_list = [["ball", "hand_right1"], ["ball", "hand_right2"]]
        adh_ids = [acts["adh_right_hand"][0], acts["adh_right_hand"][0]]
        let_go_ids = []
        let_go_times = []
    elif exp_name == "tennis_serve":
        sites = [RHAND_S, LHAND_S]  # Move
        tennis_idxs = two_arm_idxs(model)
        site_grad_idxs = [
            tennis_idxs["right_arm_without_adh"],
            tennis_idxs["left_arm_without_adh"],
        ]
        site_grad_idxs = [
            tennis_idxs["right_arm_without_adh"],
            tennis_idxs["left_arm_without_adh"],
        ]
        stabilize_jnt_idx = tennis_idxs["not_arm_j"]
        stabilize_act_idx = tennis_idxs["not_arm_a"]
        contact_check_list = [
            ["racket_handle", "hand_right1"],
            ["racket_handle", "hand_right2"],
            ["ball", "hand_left1"],
            ["ball", "hand_left2"],
        ]
        acts = opt_utils.get_act_ids(model)
        adh_ids = [
            acts["adh_right_hand"][0],
            acts["adh_right_hand"][0],
            acts["adh_left_hand"][0],
            acts["adh_left_hand"][0],
        ]
        act_ids = ["adh_right_hand", "adh_right_hand", "adh_left_hand", "adh_left_hand"]
        let_go_ids = [acts["adh_left_hand"][0]]
    elif exp_name == "tennis_grab":
        sites = [RHAND_S, LHAND_S]
        tennis_idxs = two_arm_idxs(model)
        site_grad_idxs = [
            tennis_idxs["right_arm_without_adh"],
            tennis_idxs["left_arm_without_adh"],
        ]
        stabilize_jnt_idx = tennis_idxs["not_arm_j"]
        stabilize_act_idx = tennis_idxs["not_arm_a"]

        contact_check_list = [
            ["racket_handle", "hand_right1"],
            ["racket_handle", "hand_right2"],
            ["ball", "hand_left1"],
            ["ball", "hand_left2"],
        ]
        adh_ids = [
            acts["adh_right_hand"][0],
            acts["adh_right_hand"][0],
            acts["adh_left_hand"][0],
            acts["adh_left_hand"][0],
        ]
        act_ids = ["adh_right_hand", "adh_right_hand", "adh_left_hand", "adh_left_hand"]
        let_go_ids = []
        let_go_times = []

    out_dict = dict(
        sites=sites,
        site_grad_idxs=site_grad_idxs,
        stabilize_jnt_idx=stabilize_jnt_idx,
        stabilize_act_idx=stabilize_act_idx,
        free_act_idx=other_act_idx,
        # free_act_idx=
        contact_check_list=contact_check_list,
        adh_ids=adh_ids,
        let_go_ids=let_go_ids,
    )
    return out_dict


def get_times(env, exp_name, Tf):
    model = env.model
    data = env.data
    dt = model.opt.timestep
    Tk = int(Tf / dt)
    time_dict = None
    grab_tk = 0
    let_go_times = []
    if exp_name == "basic_movements_right":
        pass
    elif exp_name == "throw_ball":
        time_dict = throw_traj(model, data, Tk)[-1]
        grab_t = Tf / 2.2
        grab_tk = int(grab_t / dt)
        let_go_times = [Tk]
    elif exp_name == "grab_ball":
        out = throw_grab_traj(model, data, Tk)
        time_dict = out[1]
        grab_time = int(time_dict["t_1"] * 0.9)
        grab_t = Tf / 2.2
        grab_tk = int(grab_t / dt)
    elif exp_name == "tennis_serve":
        time_dict = tennis_traj(model, data, Tk)[-1]
        grab_t = Tf / 2.8
        grab_tk = int(grab_t / dt)
        let_go_times = [time_dict["t_left_3"]]
    elif exp_name == "tennis_grab":
        time_dict = tennis_traj(model, data, Tk)[-1]
        grab_t = Tf / 2.2
        grab_tk = int(grab_t / dt)
    out_dict = dict(grab_phase_tk=grab_tk, let_go_times=let_go_times)
    return out_dict


def get_data_from_qtarg_file(file_loc, dt=None):
    file_conts = []
    with open(file_loc, "r") as fid:
        for line in fid:
            file_conts.append(line.split("|"))
    joint_names = file_conts[0][1:]
    joint_names[-1] = joint_names[-1].strip("\n")  # Remove \n character
    joint_names = [j.strip(" ") for j in joint_names]
    q_pos_targs = {}
    q_data_time_tks = []
    for row in file_conts[1:]:
        vals = [float(x.strip(" ").strip("\n")) for x in row[1:]]
        tv = float(row[0].strip(" "))
        if dt is not None:
            tk = int(tv / dt)
        else:
            tk = tv
        q_pos_targs[tk] = vals
        q_data_time_tks.append(tk)
    return q_pos_targs, q_data_time_tks, joint_names


def make_traj_sets(env, exp_name, Tk, t_incr, incr_every, max_its, seed=2):
    model = env.model
    data = env.data
    # smoothing_sigma = int(.1 / model.opt.timestep)
    # arc_std = 0.0001 / model.opt.timestep
    arc_std = 0.02
    # smoothing_time = 0.1
    smoothing_time = 0.2
    joints = opt_utils.get_joint_ids(model)
    left_arm_dofadr = joints["body"]["left_arm_dofadrs"]
    # right_arm_dofadr = opt_utils.convert_dofadr(
    # model, None, joints['body']['right_arm'], True)
    right_arm_dofadr = joints["body"]["right_arm_dofadrs"]
    dof_offset = model.nq - model.nv
    # TODO: check
    # left_arm_vel_id = [x+model.nq-dof_offset for x in left_arm_dofadr]
    # left_arm_vel_id = [x + model.nv for x in left_arm_dofadr]
    # right_arm_vel_id = [x + model.nv for x in right_arm_dofadr]
    acts = opt_utils.get_act_ids(model)
    # q_targ = np.zeros((Tk, 2*model.nq))
    out_idx = get_idx_sets(env, exp_name)
    syssize = model.nq + model.nv
    syssize2 = 2 * model.nv
    # TODO: qposadr versus id versus dofadr
    # t_incr = params["t_incr"]
    dt = model.opt.timestep
    amnt_to_incr = int(t_incr / dt)
    incr_time_left_endpoints = list(range(0, Tk, amnt_to_incr))
    max_incr_its = len(incr_time_left_endpoints)
    incr_it_left_endpoints = list(range(0, max_incr_its * incr_every, incr_every))

    targ_traj_masks = masks.make_basic_xpos_masks(incr_time_left_endpoints, Tk)
    targ_traj_mask_dict = {
        incr_it_left_endpoints[k]: mask for k, mask in enumerate(targ_traj_masks)
    }

    def get_qpos_data(joint_targs_file):
        q_pos_targs, q_data_time_tks, joint_names = get_data_from_qtarg_file(
            joint_targs_file, dt
        )
        q_pos_opt_ids = [model.joint(n).dofadr.item() for n in joint_names]
        q_pos_masks = masks.make_basic_qpos_masks(
            q_data_time_tks,
            q_pos_opt_ids,
            incr_time_left_endpoints,
            model.nq,
            Tk,
        )
        q_pos_mask_dict = {incr_its[k]: mask for k, mask in enumerate(q_pos_masks)}
        q_vel_mask = np.zeros((Tk, model.nv))
        q_vel_mask_dict = {incr_it: q_vel_mask for incr_it in incr_its}
        q_vel_targs = {}
        return (
            q_pos_targs,
            q_vel_targs,
            q_pos_mask_dict,
            q_vel_mask_dict,
            q_pos_opt_ids,
        )

    # def get_traj_mask()

    def make_return_dict(
        targ_trajs,
        targ_traj_masks,
        q_pos_targs,
        q_vel_targs,
        q_pos_masks,
        q_vel_masks,
        ctrl_reg_weights,
    ):
        return dict(
            targ_trajs=targ_trajs,
            targ_traj_masks=targ_traj_masks,
            targ_traj_mask_types=mask_types,
            q_pos_targs=q_pos_targs,
            q_vel_targs=q_vel_targs,
            q_pos_masks=q_pos_masks,
            q_vel_masks=q_vel_masks,
            ctrl_reg_weights=ctrl_reg_weights,
        )

    if exp_name == "basic_movements_right":
        joint_targs_file = "exp_configs/basic_movements_right_joint_targs.csv"
        (
            q_pos_targs,
            q_vel_targs,
            q_pos_masks,
            q_vel_masks,
            q_pos_opt_ids,
        ) = get_qpos_data(joint_targs_file)
        breakpoint()

        rs, thetas, wrist_qs = basic_movements.random_arcs_right_arm(
            model, data, Tk, data.site(RHAND_S).xpos, smoothing_time, arc_std, seed
        )
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:, 1] = rs * np.cos(thetas)
        traj1_xs[:, 2] = rs * np.sin(thetas)
        traj1_xs += data.site(RSHOULD_S).xpos
        targ_trajs = traj1_xs
        ctrl_reg_weights = [None]
        targ_traj_masks = masks.make_basic_xpos_masks(incr_time_right_endpoints, Tk)
        return make_return_dict(
            targ_trajs,
            targ_traj_masks,
            q_pos_targs,
            q_vel_targs,
            q_pos_masks,
            q_vel_masks,
            ctrl_reg_weights,
        )
        breakpoint()
    elif exp_name == "basic_movements_left":
        rs, thetas, wrist_qs = basic_movements.random_arcs_left_arm(
            model, data, Tk, data.site(LHAND_S).xpos, smoothing_time, arc_std, seed
        )
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:, 1] = rs * np.cos(thetas)
        traj1_xs[:, 2] = rs * np.sin(thetas)
        traj1_xs += data.site(LSHOULD_S).xpos
        full_traj = traj1_xs
        targ_traj_mask_dict = np.ones((Tk,))
        targ_traj_mask_type = "double_sided_progressive"
        # plt.plot(full_traj[:,1])
        # plt.plot(full_traj[:,2])
        # plt.show()

        targ_trajs = [full_traj]
        targ_traj_masks = [targ_traj_mask_dict]
        mask_types = [targ_traj_mask_type]

        q_targs = [np.zeros((Tk, syssize))]
        q_targ_mask = np.zeros((Tk, syssize2))
        q_targ_mask[:, left_arm_vel_id] = 1
        q_targ_masks = [q_targ_mask]
        q_targ_mask_types = ["const"]
        ctrl_reg_weights = [None]
        breakpoint()
    elif exp_name == "basic_movements_both":
        rs, thetas, wrist_qs = basic_movements.random_arcs_right_arm(
            model, data, Tk, data.site(RHAND_S).xpos, smoothing_time, arc_std
        )
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:, 1] = rs * np.cos(thetas)
        traj1_xs[:, 2] = rs * np.sin(thetas)
        traj1_xs += data.site(RSHOULD_S).xpos
        full_traj = traj1_xs
        targ_traj_mask_dict = np.ones((Tk,))
        targ_traj_mask_type = "double_sided_progressive"

        targ_trajs = [full_traj]
        targ_traj_masks = [targ_traj_mask_dict]
        mask_types = [targ_traj_mask_type]

        rs, thetas, wrist_qs = basic_movements.random_arcs_left_arm(
            model, data, Tk, data.site(LHAND_S).xpos, smoothing_time, arc_std
        )
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:, 1] = rs * np.cos(thetas)
        traj1_xs[:, 2] = rs * np.sin(thetas)
        traj1_xs += data.site(LSHOULD_S).xpos
        full_traj = traj1_xs
        targ_traj_mask_dict = np.ones((Tk,))
        targ_traj_mask_type = "double_sided_progressive"
        # plt.plot(full_traj[:,1])
        # plt.show()

        targ_trajs += [full_traj]
        targ_traj_masks += [targ_traj_mask_dict]
        mask_types = ["double_sided_progressive", "double_sided_progressive"]

        q_targs = [np.zeros((Tk, model.nq)), np.zeros((Tk, model.nq))]
        q_targ_masks = [np.zeros((Tk, model.nq)), np.zeros((Tk, model.nq))]
        q_targ_mask_types = ["const", "const"]
        ctrl_reg_weights = [None]
    elif exp_name == "throw_ball":
        targ_traj_mask_dict = np.ones((Tk,))
        targ_traj_mask_type = "double_sided_progressive"
        out = throw_traj(model, data, Tk)
        full_traj, time_dict = out

        bodyj = joints["body"]["body_dofs"]

        targ_trajs = [full_traj]
        targ_traj_masks = [targ_traj_mask_dict]
        mask_types = [targ_traj_mask_type]

        q_targs = [np.zeros((Tk, syssize))]
        q_targ_mask = np.zeros((Tk, syssize2))
        q_targ_mask2 = np.zeros((Tk, syssize2))
        # TODO: resolve quaternion
        q_targ_mask2[time_dict["t_1"] :, joints["all"]["wrist_left"]] = 1
        q_targ_nz = np.linspace(0, -2.44, time_dict["t_2"] - time_dict["t_1"])
        q_targ[time_dict["t_1"] : time_dict["t_2"], joints["all"]["wrist_left"]] = (
            q_targ_nz
        )
        q_targ[time_dict["t_2"] :, joints["all"]["wrist_left"]] = -2.44
        q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask, q_targ_mask]
        q_targ_mask_types = ["const"]
        q_targs = [q_targ]
        ctrl_reg_weights = [None]
    elif exp_name == "grab_ball":
        targ_traj_mask_dict = np.ones((Tk,))
        # targ_traj_mask_type = 'progressive'
        targ_traj_mask_type = "double_sided_progressive"
        out = throw_grab_traj(model, data, Tk)
        full_traj, time_dict = out
        contact_check_list = [["ball", "hand_right1"], ["ball", "hand_right2"]]
        adh_ids = [acts["adh_right_hand"][0], acts["adh_right_hand"][0]]
        let_go_ids = []
        # let_go_times = [Tk]
        let_go_times = []
        targ_trajs = [full_traj]
        targ_traj_masks = [targ_traj_mask_dict]
        mask_types = [targ_traj_mask_type]

        q_targs = [np.zeros((Tk, syssize))]
        q_targ_mask = np.zeros((Tk, syssize))
        q_targ_mask2 = np.zeros((Tk, syssize))
        q_targ_mask2[time_dict["t_1"] :, joints["all"]["wrist_left"]] = 1
        q_targ_nz = np.linspace(0, -2.44, time_dict["t_2"] - time_dict["t_1"])
        q_targ[time_dict["t_1"] : time_dict["t_2"], joints["all"]["wrist_left"]] = (
            q_targ_nz
        )
        q_targ[time_dict["t_2"] :, joints["all"]["wrist_left"]] = -2.44
        q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask, q_targ_mask]
        q_targ_mask_types = ["const"]
        q_targs = [q_targ]
        ctrl_reg_weights = [None]
    elif exp_name == "tennis_serve":
        targ_traj_mask_dict = np.ones((Tk,))
        # targ_traj_mask_type = 'progressive'
        targ_traj_mask_type = "double_sided_progressive"
        # targ_traj_mask_type = 'const'
        out = tennis_traj(model, data, Tk)
        right_hand_traj, left_hand_traj, ball_traj, time_dict = out
        ball_traj_mask = np.ones((Tk,))
        ball_traj_mask[time_dict["t_left_3"] :] = 0
        out = tennis_traj(model, data, Tk)
        right_hand_traj, left_hand_traj, ball_traj, time_dict = out
        targ_trajs = [right_hand_traj, left_hand_traj]
        targ_traj_masks = [targ_traj_mask_dict, targ_traj_mask_dict]
        mask_types = [targ_traj_mask_type] * 2
        # q_targ = np.zeros((Tk, 2*model.nq))
        bot = 0.6
        q_targ = np.ones((Tk, syssize)) * bot
        q_targ_mask = np.zeros((Tk, syssize))
        q_targ_mask2 = np.zeros((Tk, syssize))
        # q_targ_mask2[time_dict['t_left_1']:time_dict['t_left_3'],
        # joints['all']['wrist_left']] = 1
        # tp = int(time_dict['t_left_1'] / 2)
        tp = time_dict["t_left_1"]
        q_targ_mask2[tp : time_dict["t_left_3"], joints["all"]["wrist_left"]] = 1
        tmp = np.linspace(0, 1, time_dict["t_left_3"] - time_dict["t_left_1"])
        tmp = sigmoid(tmp, 5)
        # bot = .75
        q_targ_nz = (2.3 - bot) * tmp + bot
        # tmp = np.linspace(.75, 2.3, time_dict['t_left_2']-time_dict['t_left_1'])
        # q_targ_nz = sigmoid(tmp, 3)
        q_targ[
            time_dict["t_left_1"] : time_dict["t_left_3"], joints["all"]["wrist_left"]
        ] = q_targ_nz
        # q_targ[time_dict['t_left_2']:, joints['all']['wrist_left']] = 2.3
        q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask]
        q_targ_mask_types = ["const"] * 2
        q_targs = [q_targ] * 2

        ctrl_reg_weight = np.ones((Tk - 1, len(out_idx["site_grad_idxs"][1])))
        ctrl_reg_weight[:, -1] = 100
        ctrl_reg_weights = [None, ctrl_reg_weight]
    elif exp_name == "tennis_grab":
        targ_traj_mask_dict = np.ones((Tk,))
        targ_traj_mask_type = "double_sided_progressive"
        out = tennis_grab_traj(model, data, Tk)
        right_hand_traj, left_hand_traj, ball_traj, time_dict = out
        targ_trajs = [right_hand_traj, left_hand_traj]
        targ_traj_masks = [targ_traj_mask_dict, targ_traj_mask_dict]
        mask_types = [targ_traj_mask_type] * 2
        q_targs = [np.zeros((Tk, syssize))]
        q_targ_mask = np.zeros((Tk, syssize))
        q_targ_mask2 = np.zeros((Tk, syssize))
        q_targ_mask2[time_dict["t_left_1"] :, joints["all"]["wrist_left"]] = 1
        q_targ_nz = np.linspace(0, -2.44, Tk - time_dict["t_left_1"])
        q_targ[time_dict["t_left_1"] :, joints["all"]["wrist_left"]] = q_targ_nz
        q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask]
        q_targ_mask_types = ["const"] * 3
        q_targs = [q_targ] * 3
        ctrl_reg_weights = [None] * 3
        # plt.plot(right_hand_traj[:,1])
        # plt.plot(right_hand_traj[:,2])
        # plt.show()

    return out_dict


def forward_and_collect_data(env, ctrls, ret_fn=None, render=False):
    """Simulate and collect data with ret_fn. ret_fn will take
    data as an input argument and output a dictionary, and it is called
    at every timestep."""
    model = env.model
    data = env.data
    if callable(render):
        render_fn = render
        render = True
    elif render:
        render_fn = env.render
    else:
        render_fn = lambda: None

    ret_vals = []
    Tk = ctrls.shape[0]
    if ret_fn is not None:
        ret_vals.append(ret_fn(data))
    render_fn()
    for tk in range(Tk):
        util.step(model, data, ctrls[tk])
        if ret_fn is not None:
            ret_vals.append(ret_fn(data))
        render_fn()
    if ret_fn is not None:  # Now switch the key and time axes of ret_vals
        dict_keys = ret_vals[0].keys()
        ret_dict = {
            key: np.zeros((Tk + 1, len(val))) for key, val in ret_vals[0].items()
        }
        for tk in range(Tk + 1):
            for key in dict_keys:
                ret_dict[key][tk] = ret_vals[tk][key]
        return ret_dict


def forward_with_dynamic_adhesion(
    env,
    ctrls,
    noisev=None,
    render=True,  # Can also be a callable (function) which will be called to render
    let_go_times=[],
    let_go_ids=[],
    n_steps_adh=10,
    contact_check_list=[],
    adh_ids=[],
):
    """Simulate dynamics according to ctrls, while performing dynamic adhesion. This adhesion will automatically ramp up adhesion actuators when
    contacts as defined by contact_check_list are detected."""
    model = env.model
    data = env.data
    if callable(render):
        render_fn = render
        render = True
    else:
        render_fn = env.render
    Tk = ctrls.shape[0]
    adh_ctrl = opt_utils.AdhCtrl(
        let_go_times, let_go_ids, n_steps_adh, contact_check_list, adh_ids
    )
    if noisev is None:
        noisev = np.zeros((Tk, model.nu))
    for k in range(Tk):
        ctrls[k], _, _ = adh_ctrl.get_ctrl(model, data, ctrls[k])
        util.step(model, data, ctrls[k] + noisev[k])
        if render:
            render_fn()
    return ctrls


class LimLowestDict:
    def __init__(self, max_len):
        self.max_len = max_len
        self.dict = sc.SortedDict()

    def append(self, key, val):
        self.dict.update({key: val})
        if len(self.dict) > self.max_len:
            self.dict.popitem()


class DoubleSidedProgressive:
    def __init__(
        self,
        incr_every,
        amnt_to_incr,
        grab_phase_it,
        grab_phase_tk,
        phase_2_it,
        max_idx=1e8,
    ):
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
        self.phase = "grab"

    def _update_grab_phase(self):
        start_idx = 0
        end_idx = self.grab_phase_tk
        self.grab_end_idx = end_idx
        return slice(start_idx, end_idx)

    def _update_phase_1(self):
        start_idx = self.grab_end_idx
        end_idx = self.amnt_to_incr * (self.incr_cnt + 1) + start_idx
        idx = slice(start_idx, end_idx)
        self.incr_cnt += 1
        return idx

    def _update_phase_2(self):
        start_idx = self.amnt_to_incr * self.incr_cnt2 + self.grab_end_idx
        end_idx = self.amnt_to_incr * (self.incr_cnt + 1) + start_idx
        self.incr_cnt2 += 1
        idx = slice(start_idx, end_idx)
        return idx

    def update(self):
        if self.k < self.grab_phase_it:
            self.idx = self._update_grab_phase()
        elif self.k == self.grab_phase_it:
            self.phase = "phase_1"
            self.incr_k = 0
        if self.phase != "grab" and self.incr_k % self.incr_every == 0:
            if self.k >= self.phase_2_it:
                self.phase = "phase_2"
                self.idx = self._update_phase_2()
            else:
                self.phase = "phase_1"
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
        end_idx = self.amnt_to_incr * (self.incr_cnt + 1)
        start_idx = max(0, end_idx - self.window_size)
        if self.k % self.incr_every == 0:
            self.idx = slice(start_idx, end_idx)
            self.incr_cnt += 1
        self.k += 1
        return self.idx


def show_plot(
    axs,
    hxs,
    tt,
    target_trajs,
    targ_traj_mask,
    site_names=None,
    site_grad_idxs=None,
    ctrls=None,
    grads=None,
    qvals=None,
    qtargs=None,
    show=True,
    save=False,
):
    fig = axs[0, 0].figure
    n = len(hxs)
    nr = range(n)
    ax_cntr = 0
    for k in nr:
        hx = hxs[k]
        tm = np.tile(targ_traj_mask[k], (3, 1)).T
        tm[tm == 0] = np.nan
        ft = target_trajs[k] * tm
        loss = np.mean((hx - target_trajs[k]) ** 2)
        ax = axs[0, k]
        ax.cla()
        ax.plot(tt, hx[:, 1], color="blue", label="x")
        ax.plot(tt, ft[:, 1], "--", color="blue")
        ax.plot(tt, hx[:, 2], color="red", label="y")
        ax.plot(tt, ft[:, 2], "--", color="red")
        lims = ax.get_xlim()
        if site_names is not None:
            ax.set_title(site_names[k])
        ax.legend()
    ax_cntr += 1
    if ctrls is not None and site_grad_idxs is not None:
        for k in nr:
            ax = axs[ax_cntr, k]
            ax.cla()
            ax.plot(tt[:-1], ctrls[:, site_grad_idxs[k]])
        ax_cntr += 1
    if grads is not None:
        for k in nr:
            ax = axs[ax_cntr, k]
            ax.cla()
            grad = np.zeros((len(tt) - 1, grads[0].shape[1]))
            grad[: grads[k].shape[0]] = grads[k]
            ax.plot(tt[:-1], grad)
        ax_cntr += 1
    # if ctrls is not None:
    # axs[1,0].plot(tt[:-1], ctrls[:, -2])
    # axs[1,1].plot(tt[:-1], ctrls[:, -1])
    if qvals is not None:
        for k in nr:
            ax = axs[ax_cntr, k]
            ax.cla()
            ax.plot(tt, qvals[k])
            ax.set_prop_cycle(None)
            ax.plot(tt, qtargs[k], "--")
            ax.set_xlim(lims)
    fig.tight_layout()
    if show:
        plt.show(block=False)
        # plt.show(block=True)
        plt.pause(0.05)
    if save:
        fig.savefig("fig.pdf")


def get_last_timepoint(mask):
    """Get index of last nonzero entry in mask."""
    return np.where(mask)[0][-1].item()


class targetRender:
    def __init__(self, env, target_data_list, sites) -> None:
        self.counter = 0
        self.target_data_list = target_data_list
        self.env = env
        self.sites = sites

    def render(self):
        for k, target_data in enumerate(self.target_data_list):
            marker_pos = target_data[self.counter]
            self.env.mujoco_renderer.viewer.add_marker(
                size=np.array([0.05, 0.05, 0.05]),
                pos=marker_pos,
                matid=0,
                rgba=(1, 1, 0, 1),
                type=mj.mjtGeom.mjGEOM_SPHERE,
                label="targ",
                emission=0,
                specular=0.5,
                shininess=0.5,
                reflectance=0,
            )
            # breakpoint()
            marker_pos = self.env.data.site(self.sites[k]).xpos
            self.env.mujoco_renderer.viewer.add_marker(
                size=np.array([0.05, 0.05, 0.05]),
                pos=marker_pos,
                matid=0,
                rgba=(1, 1, 0, 1),
                type=mj.mjtGeom.mjGEOM_SPHERE,
                label="hand",
                emission=0,
                specular=0.5,
                shininess=0.5,
                reflectance=0,
            )
        self.env.render()
        self.counter += 1

    def reset_counter(self):
        self.counter = 0


def get_from_interv_dict(interv_dict: dict[int | float, Any], lookup_idx: int | float):
    """
    If interv_dict = {0: 'A', 30: 'B', 50: 'C'} then:
        key = 0 -> return 'A'
        key = 10 -> return 'A'
        key = 30 -> return 'B'
        key = 60 -> return 'C'
    """
    dkeys = list(interv_dict.keys())
    dkeys = sorted(dkeys)
    key = -1
    prev_key = dkeys[0]
    for key in dkeys:
        if lookup_idx < key:
            break
        prev_key = key
    return interv_dict[prev_key]


def arm_target_traj(
    env,
    sites,
    site_grad_idxs,
    stabilize_jnt_idx,
    stabilize_act_idx,
    targ_trajs,
    targ_traj_masks: Dict,
    targ_traj_mask_types,
    q_targs,
    q_pos_targ_masks,
    q_vel_targ_masks,
    ctrls,
    grad_trunc_tk,
    seed,
    ctrl_rate,
    ctrl_std,
    Tk,
    max_its=30,
    lr=10,
    lr2=10,
    it_lr2=31,
    keep_top=1,
    incr_every=5,
    amnt_to_incr=5,
    grad_update_every=1,
    grab_phase_it=0,
    grab_phase_tk=0,
    phase_2_it=None,
    plot_every=1,
    render_every=1,
    optimizer="adam",
    contact_check_list=[],
    adh_ids=[],
    balance_cost=1000,
    joint_cost=100,
    root_cost=0,
    foot_cost=1000,
    ctrl_cost=1,
    let_go_times=[],
    let_go_ids=[],
    n_steps_adh=10,
    ctrl_reg_weights=None,
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
        targ_traj_masks: dict of target trajectory masks
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
        phase_2_it = max_its
    if plot_every is None:
        update_plot_every = max_its
    if render_every is None:
        render_every = max_its
    if ctrl_reg_weights is None:
        ctrl_reg_weights = [None] * len(site_names)

    model = env.model
    data = env.data
    nq = model.nq

    incr_its = sorted(list(targ_traj_masks.keys()))

    render_class = targetRender(env, targ_trajs, sites)
    render_fn = render_class.render

    not_stabilize_act_idx = [k for k in range(model.nu) if k not in stabilize_act_idx]

    n_sites = len(sites)
    assert (
        n_sites == len(targ_trajs)
        and n_sites == len(targ_traj_masks)
        and n_sites == len(targ_traj_mask_types)
    )

    data0 = copy.deepcopy(data)

    noisev = make_noisev(model, seed, Tk, ctrl_std, ctrl_rate)

    util.reset_state(model, data, data0)

    def ret_fn(data):
        site_dict = {}
        for site in sites:
            site_dict[site] = data.site(site).xpos.copy()
        site_dict.update(
            {
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
            }
        )
        return site_dict

    # def ret_fn(data):
    #     return {
    #         "qpos": data.qpos.copy(),
    #         "qvel": data.qvel.copy(),
    #         "sensordata": data.sensordata.copy(),
    #     }

    ### Gradient descent
    qpos0 = data.qpos.copy()

    dt = model.opt.timestep
    T = Tk * dt
    tt = np.arange(0, T, dt)

    joints = opt_utils.get_joint_ids(model)

    progbar = util.ProgressBar(final_it=max_its)

    def get_opt(lr):
        if optimizer == "rmsprop":
            return opts.RMSProp(lr=lr)
        if optimizer == "adam":
            return opts.Adam(lr=lr)
        if optimizer == "mom_sgd":
            return opts.SGD(lr=lr, momentum=0.2)
        if optimizer == "sgd":
            return opts.SGD(lr=lr, momentum=0.2)

    optms = []
    for k in range(n_sites):
        optms.append(get_opt(lr))
    lowest_losses = LimLowestDict(keep_top)
    lowest_losses_curr_mask = LimLowestDict(keep_top)

    dq = np.zeros(model.nv)
    for k0 in range(max_its):
        if k0 >= it_lr2:
            lr = lr2
        if k0 in incr_its:
            for k in range(n_sites):
                optms[k] = get_opt(lr)
        progbar.update(" it: " + str(k0))
        targ_traj_mask_curr = get_from_interv_dict(targ_traj_masks, k0)
        Tk_trunc = get_last_timepoint(targ_traj_mask_curr)
        ctrls_trunc = ctrls[:Tk_trunc]
        noisev_trunc = noisev[:Tk_trunc]
        util.reset_state(model, data, data0)
        ctrls_trunc = forward_with_dynamic_adhesion(
            env,
            ctrls_trunc,
            noisev_trunc,
            False,
            let_go_times,
            let_go_ids,
            n_steps_adh,
            contact_check_list,
            adh_ids,
        )
        util.reset_state(model, data, data0)
        grads = [0] * n_sites
        update_phase = k0 % grad_update_every
        for k in range(n_sites):
            grads[k] = opt_utils.traj_deriv_new(
                model,
                data,
                ctrls_trunc + noisev_trunc,
                targ_trajs[k][: Tk_trunc + 1],
                targ_traj_mask_curr[: Tk_trunc + 1],
                q_targs[k][: Tk_trunc + 1],
                q_targ_masks[k][: Tk_trunc + 1],
                grad_trunc_tk,
                deriv_ids=site_grad_idxs[k],
                deriv_site=sites[k],
                update_every=grad_update_every,
                update_phase=update_phase,
                let_go_times=let_go_times,
                let_go_ids=let_go_ids,
                n_steps_adh=n_steps_adh,
                contact_check_list=contact_check_list,
                adh_ids=adh_ids,
                ctrl_reg_weight=ctrl_reg_weights[k],
            )
            util.reset_state(model, data, data0)
        losses = [0] * n_sites
        for k in range(n_sites):
            ctrls_trunc[:, site_grad_idxs[k]] = optms[k].update(
                ctrls_trunc[:, site_grad_idxs[k]], grads[k], "ctrls", losses[k]
            )

        try:
            ctrls_trunc, __, qs, qvels = opt_utils.get_stabilized_ctrls(
                model,
                data,
                Tk_trunc + 1,
                noisev_trunc,
                qpos0,
                stabilize_act_idx,
                stabilize_jnt_idx,
                ctrls_trunc[:, not_stabilize_act_idx],
                K_update_interv=10000,
                balance_cost=balance_cost,
                joint_cost=joint_cost,
                root_cost=root_cost,
                foot_cost=foot_cost,
                ctrl_cost=ctrl_cost,
                let_go_times=let_go_times,
                let_go_ids=let_go_ids,
                n_steps_adh=n_steps_adh,
            )
            # print("Testing...")
            # loop = 'r'
            # while loop == 'r':
            # util.reset_state(model, data, data0)
            # env.reset_sim_time_counter()
            # util.forward_sim_render(env, ctrls)
            # loop = input("Enter 'r' to rerun simulation: ")
        except np.linalg.LinAlgError:
            print("LinAlgError in get_stabilized_ctrls")
            ctrls_trunc[:, not_stabilize_act_idx] *= 0.99
        ctrls[:Tk_trunc] = ctrls_trunc.copy()
        # ctrls = np.clip(ctrls, -1, 1)
        if True:
            tk = Tk_trunc
        else:
            tk = Tk
        util.reset_state(model, data, data0)
        render = k0 % render_every == 0
        if render:
            ret_dict = forward_and_collect_data(env, ctrls[:tk], ret_fn, render_fn)
            render_class.reset_counter()
        else:
            ret_dict = forward_and_collect_data(env, ctrls[:tk], ret_fn, False)
        # hxs, qs = forward_with_sites(env, ctrls[:tk], sites, render=False)
        qs = ret_dict["qpos"]
        qvs = ret_dict["qvel"]
        q_targs_masked = []
        qs_list = []
        hxs = [0] * n_sites
        dldss = [0] * n_sites
        losses_curr_mask = [0] * n_sites
        for k in range(n_sites):
            hx = ret_dict[sites[k]]
            hxs[k] = hx
            # hx = hxs[k]
            # qs_k = qs * q_targ_masks[k]
            # q_targ = q_targs[k] * q_targ_masks[k]
            # diffsq2 =  (qs_k - q_targ)**2
            diffsq1 = (hx - targ_trajs[k][: tk + 1]) ** 2
            mask = q_targ_masks[k][: tk + 1]
            nonzero = np.sum(mask > 0)
            if nonzero > 0:
                qs_k = qs.copy()
                qvs_k = qvs.copy()
                q_targ = q_targs[k][: tk + 1]
                dq = opt_utils.batch_differentiatePos(
                    model,
                    1,
                    qs_k * mask[:, :nq],
                    q_targ[:, :nq] * mask[:, :nq],
                )
                dvel = (qvs_k - q_targ[:, nq:]) * mask[:, nq:]
                dqfull = np.concatenate((dq, dvel))
                diffsq2 = dqfull**2
                sum2 = np.sum(diffsq2) / np.sum(mask > 0)
            else:
                sum2 = 0
            # losses[k] = np.mean(diffsq1) + sum2
            losses[k] = np.mean(diffsq1)
            mask = np.tile((targ_traj_mask_curr[: tk + 1] > 0), (3, 1)).T
            temp = np.sum(diffsq1 * mask) / (np.sum(mask[:, 0]))
            losses_curr_mask[k] = temp

            q_targs_masked_tmp = q_targs[k][: tk + 1].copy()
            q_targs_masked_tmp[q_targ_masks[k][: tk + 1] == 0] = np.nan
            q_targs_masked.append(q_targs_masked_tmp)
            qs_tmp = qs.copy()
            qs_tmp[q_targ_masks[k][: tk + 1, :nq] == 0] = np.nan
            qs_list.append(qs_tmp)
        loss = sum([loss.item() for loss in losses]) / n_sites
        lowest_losses.append(loss, (k0, ctrls.copy()))
        loss_curr_mask_avg = sum([loss.item() for loss in losses_curr_mask]) / n_sites
        lowest_losses_curr_mask.append(loss_curr_mask_avg, (k0, ctrls.copy()))
        toc = time.time()
        # print(loss, toc-tic)

        nr = range(n_sites)
        if k0 % plot_every == 0:
            # qs_wr = qs[:, joints['all']['wrist_left']]
            # print()
            # print(ctrls[:10, :5])
            # print()
            # print(ctrls_trunc[:10, :5])
            # print()
            # print(grads[0][:10, :5])
            # print()
            # breakpoint()
            show_plot(
                axs,
                hxs,
                tt[: tk + 1],
                [x[: tk + 1] for x in targ_trajs],
                [x[: tk + 1] for x in targ_traj_mask_currs],
                # qs_wr,
                # q_targs_wr,
                sites,
                site_grad_idxs,
                ctrls[:tk],
                grads,
                qs_list,
                q_targs_masked,
                show=True,
                save=False,
            )
            plt.pause(0.1)
            if k0 == 0:
                # Plot again to refresh the window so it resizes to a proper size
                show_plot(
                    axs,
                    hxs,
                    tt[: tk + 1],
                    [x[: tk + 1] for x in targ_trajs],
                    [x[: tk + 1] for x in targ_traj_mask_currs],
                    # qs_wr,
                    # q_targs_wr,
                    sites,
                    site_grad_idxs,
                    ctrls[:tk],
                    grads,
                    qs_list,
                    q_targs_masked,
                    show=False,
                    save=True,
                )
                plt.pause(0.1)
        # util.reset_state(model, data, data0)
        # ctrls = forward_with_dynamic_adhesion(env, ctrls, noisev, True)
        # plt.show()
        # if k0 > phase_2_it:

        # util.reset_state(model, data, data0)
        # hx = forward_with_site(env, ctrls, site_names[0], True)
    # except KeyboardInterrupt:
    # pass

    return ctrls, lowest_losses.dict
