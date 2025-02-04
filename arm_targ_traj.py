from typing import Dict, List, Any
import util as butil
import opt_utils as opt_utils
import optimizers as opts
import numpy as np
import sim_util as util
import mujoco as mj
import copy
import sortedcontainers as sc
import pickle as pkl

# import seaborn as sns
from matplotlib import pyplot as plt
import time
import basic_movements
import masks

plt.style.use("tableau-colorblind10")
plt.rcParams.update({"font.size": 8})
plt.rcParams.update({"font.size": 8})
plt.rcParams.update({"axes.titlesize": "small"})

# sns.color_palette("colorblind")

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
    grab_targ = data.site("ball").xpos + np.array([0, 0, 0.05])
    s = sigmoid(np.linspace(0, 1, Tk1), 5)
    s = np.tile(s, (3, 1)).T
    grab_traj = handx + s * (grab_targ - handx)
    # grab_traj[-1] = grab_targ

    setup_traj = np.zeros((Tk2, 3))
    s = np.linspace(0, 1, Tk2 - Tk1)
    s = np.stack((s, s, s)).T
    setup_traj = grab_traj[-1] + s * (arc_traj_vs[0] - grab_traj[-1])
    traj = np.concatenate((grab_traj, setup_traj, arc_traj_vs), axis=0)
    vel = np.diff(traj, axis=0, prepend=traj[0:1]) / model.opt.timestep

    time_dict = {
        "t_1": Tk1,
        "t_2": Tk2,
        "t_3": Tk3,
        "Tk1": Tk1,
        "Tk2": Tk2 - Tk1,
        "Tk3": Tk3 - Tk2,
    }

    return traj, vel, time_dict


def tennis_grab_traj(model, data, Tk):
    shouldxr = data.site(RSHOULD_S).xpos
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

    # fig, ax = plt.subplots()
    # t_fin = Tk * model.opt.timestep
    # tt = np.linspace(0, t_fin, Tk)
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
    # arc_traj_vs2 = arc_traj(
    #     data.site(LSHOULD_S).xpos,
    #     r,
    #     0.9 * np.pi / 2,
    #     0.7 * np.pi / 2,
    #     10,
    #     density_fn="",
    # )

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

    # fig, ax = plt.subplots()
    # t_fin = Tk * model.opt.timestep
    # tt = np.linspace(0, t_fin, Tk)
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


def get_idx_sets(env, config_name):
    model = env.model
    ids = opt_utils.get_joints(model)["body"]["ids"]
    acts = opt_utils.get_act_ids(model)

    ## Joint and actuator ids
    if config_name in [
        "basic_movements_right",
        "basic_movements_left",
        "ball_throw",
        "grab_ball",
    ]:  # One-handed actions
        if config_name == "basic_movements_left":
            sites = [LHAND_S]
            arm_str = "left_arm"
        elif config_name == "basic_movements_right":
            sites = [RHAND_S]
            arm_str = "right_arm"
        else:
            sites = [RHAND_S, "R_Hand_below"]
            arm_str = "right_arm"
        not_arm_not_root = [id for id in ids["not_root"] if id not in ids[arm_str]]
        stabilize_jnt_idx = not_arm_not_root
        arm_act = acts[arm_str]
        arm_act_without_adh = [k for k in arm_act if k not in acts["adh"]]
        # Include all adhesion (including other hand)
        not_arm_act = [
            k for k in acts["all"] if k not in arm_act and k not in acts["adh"]
        ]
        site_grad_idxs = [arm_act_without_adh] * len(sites)
        stabilize_act_idx = not_arm_act
        other_act_idx = arm_act_without_adh
    elif config_name in [
        "basic_movements_both",
        "tennis_serve",
        "tennis_grab",
    ]:  # Two-handed actions
        sites = [RHAND_S, LHAND_S]
        arm_ids = ids["right_arm"] + ids["left_arm"]
        stabilize_jnt_idx = [
            id for id in ids["not_root"] if id not in arm_ids
        ]  # Not arm ids
        arm_a = acts["right_arm"] + acts["left_arm"]  # Sorting not necessary
        stabilize_act_idx = [  # Not arm or adhesion actuators
            k for k in acts["all"] if k not in arm_a and k not in acts["adh"]
        ]
        right_arm_without_adh = [k for k in acts["right_arm"] if k not in acts["adh"]]
        left_arm_without_adh = [k for k in acts["left_arm"] if k not in acts["adh"]]
        site_grad_idxs = [
            right_arm_without_adh,
            left_arm_without_adh,
        ]

    else:
        raise ValueError("Invalid config_name")

    ## Contact check list and adhesion ids
    HAND_STR_RIGHT = "R_Hand"
    HAND_STR_LEFT = "L_Hand"
    if config_name in ["ball_grab", "ball_throw"]:
        adh_ids = [
            acts["adh_right_hand"][0],
            acts["adh_right_hand"][0],
            acts["adh_right_hand"][0],
            acts["adh_right_hand"][0],
        ]
        # contact_check_list = [["ball", "hand_right1"], ["ball", "hand_right2"]]
        contact_check_list = [["ball", HAND_STR_RIGHT + str(i)] for i in range(1, 5)]
    elif config_name in ["tennis_serve", "tennis_grab"]:
        # contact_check_list = [
        contact_check_list = [
            ["ball", HAND_STR_RIGHT + str(i)] for i in range(1, 5)
        ] + [["racket_handle", HAND_STR_LEFT + str(i)] for i in range(1, 5)]
        acts = opt_utils.get_act_ids(model)
        adh_ids = [
            acts["adh_right_hand"][0],
            acts["adh_right_hand"][0],
            acts["adh_right_hand"][0],
            acts["adh_right_hand"][0],
            acts["adh_left_hand"][0],
            acts["adh_left_hand"][0],
            acts["adh_left_hand"][0],
            acts["adh_left_hand"][0],
        ]
        breakpoint()
    else:
        adh_ids = []
        contact_check_list = []
        # act_ids = ["adh_right_hand", "adh_right_hand", "adh_left_hand", "adh_left_hand"]

    ## Letting go ids
    if config_name == "ball_throw":
        let_go_ids = [acts["adh_right_hand"][0]]
    elif config_name == "tennis_serve":
        let_go_ids = [acts["adh_left_hand"][0]]
    else:
        let_go_ids = []

    out_dict = dict(
        sites=sites,
        site_grad_idxs=site_grad_idxs,
        stabilize_jnt_idx=stabilize_jnt_idx,
        stabilize_act_idx=stabilize_act_idx,
        # free_act_idx=other_act_idx,
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
    elif exp_name == "basic_movements_left":
        pass
    elif exp_name == "ball_throw":
        time_dict = throw_traj(model, data, Tk)[-1]
        grab_t = Tf / 2.2
        grab_tk = int(grab_t / dt)
        let_go_times = [Tk]
    elif exp_name == "grab_ball":
        out = throw_grab_traj(model, data, Tk)
        time_dict = out[1]
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
    """The output datastructure assumes that the number
    of timepoints where joint targets is specified is
    relatively small in number; else a different structure
    would probably be better. This also assumes that every
    joint named in the input file has exactly one degree of
    freedom."""
    file_conts = []
    with open(file_loc, "r") as fid:
        for line in fid:
            file_conts.append(line.split("|"))
    joint_names = file_conts[0][1:]
    joint_names[-1] = joint_names[-1].strip("\n")  # Remove \n character
    joint_names = [j.strip(" ") for j in joint_names]
    # q_pos_targs = {}
    q_pos_targs = []
    q_data_time_tks = []
    for row in file_conts[1:]:
        vals = [float(x.strip(" ").strip("\n")) for x in row[1:]]
        tv = float(row[0].strip(" "))
        if dt is not None:
            tk = int(tv / dt)
        else:
            tk = tv
        q_pos_targs.append(vals)
        q_data_time_tks.append(tk)
    q_pos = {"targ_val": q_pos_targs, "tk": q_data_time_tks, "joint_names": joint_names}
    return q_pos


def make_traj_sets(
    env,
    exp_name,
    Tk,
    tk_incrs,
    incr_everys,
    phase_2_it,
    mask_window_tk,
    seed=2,
    mask_decay_factor=0.9,
    grab_phase_it=0,
    grab_phase_tk=0,
):
    """
    params:
        env: Gymnasium environment.
        exp_name: Name of the experiment, corresponding with file to be loaded for
            the joint targets.
        Tk: Final time index.
        amnt_to_incr: The amount of timesteps that the mask increments every
        time it increments.
        incr_everys: The number of iterations between mask incrments.
        seed: rng seed.
        grab_phase_it: Iteration at which the grab phase ends.
        grab_phase_tk: Time index at which the grab ends.

    TODO: This would perhaps be easier to understand if there was a
    datatype for interval dictionaries with its own description.
    Returns dictionary with keys: TODO: update
        traj_targs:  Target trajectories in cartesion coordinates.
        traj_masks:  Masks for target trajectories. Has keys corresponding
            to the start point for iteration intervals, so that the mask can
            change over iterations.
        q_pos_targs:  Target trajectories for joints positions.
        q_vel_targs:  Target trajectories for joint velocities.
        q_pos_masks:  Masks for joint positions. Has keys corresponding to the
            start point for iteration intervals, so that the mask can change
            over iterations.
        q_vel_masks:  Masks for joint velocities.
        ctrl_reg_weights:  Unused TODO: address this.
    """

    model = env.model
    data = env.data
    # smoothing_sigma = int(.1 / model.opt.timestep)
    # arc_std = 0.0001 / model.opt.timestep
    arc_std = 0.02

    incr_every = incr_everys[0]
    tk_incr = tk_incrs[0]
    phase_2 = phase_2_it is not None

    # smoothing_time = 0.1
    smoothing_time = 0.2
    joints = opt_utils.get_joints(model)
    acts = opt_utils.get_act_ids(model)
    out_idx = get_idx_sets(env, exp_name)
    syssize = model.nq + model.nv
    dt = model.opt.timestep
    # incr_time_right_endpoints = list(range(amnt_to_incr, Tk + 1, amnt_to_incr))
    incr_time_right_endpoints_before = list(range(tk_incr, grab_phase_tk, tk_incr))
    if phase_2:
        m = int((phase_2_it - grab_phase_it) / incr_every)
        tk_phase_2 = grab_phase_tk + m * tk_incr
        incr_time_right_endpoints_after = list(
            range(grab_phase_tk, tk_phase_2, tk_incr)
        )
        incr_time_right_endpoints_phase_2 = list(range(tk_phase_2, Tk + 1, tk_incrs[1]))
        if incr_time_right_endpoints_phase_2[-1] != Tk:
            incr_time_right_endpoints_phase_2.append(Tk)
        max_incr_its_before = len(incr_time_right_endpoints_before)
        max_incr_its_after = len(incr_time_right_endpoints_after)
        max_incr_its_phase_2 = len(incr_time_right_endpoints_phase_2)
        incr_time_right_endpoints = (
            incr_time_right_endpoints_before
            + incr_time_right_endpoints_after
            + incr_time_right_endpoints_phase_2
        )
        incr_it_right_endpoints_before = list(
            range(incr_every, max_incr_its_before * incr_every + 1, incr_every)
        )
        incr_it_right_endpoints_after = list(
            range(
                grab_phase_it,
                max_incr_its_after * incr_every + grab_phase_it,
                incr_every,
            )
        )
        incr_it_right_endpoints_phase_2 = list(
            range(
                phase_2_it,
                max_incr_its_phase_2 * incr_everys[1] + phase_2_it,
                incr_everys[1],
            )
        )
        incr_it_right_endpoints = (
            incr_it_right_endpoints_before
            + incr_it_right_endpoints_after
            + incr_it_right_endpoints_phase_2
        )
    else:
        incr_time_right_endpoints_after = list(range(grab_phase_tk, Tk + 1, tk_incr))
        if incr_time_right_endpoints_after[-1] != Tk:
            incr_time_right_endpoints_after.append(Tk)

        max_incr_its_before = len(incr_time_right_endpoints_before)
        max_incr_its_after = len(incr_time_right_endpoints_after)
        incr_time_right_endpoints = (
            incr_time_right_endpoints_before + incr_time_right_endpoints_after
        )
        incr_it_right_endpoints_before = list(
            range(incr_every, max_incr_its_before * incr_every + 1, incr_every)
        )
        incr_it_right_endpoints_after = list(
            range(
                grab_phase_it,
                max_incr_its_after * incr_every + grab_phase_it,
                incr_every,
            )
        )
        incr_it_right_endpoints = (
            incr_it_right_endpoints_before + incr_it_right_endpoints_after
        )
    targ_traj_mask_lists = masks.make_basic_xpos_masks(
        incr_time_right_endpoints, mask_decay_factor
    )
    targ_traj_masks = {
        incr_it_right_endpoints[k]: mask for k, mask in enumerate(targ_traj_mask_lists)
    }
    # TODO: fix case where grab_phase_it is less than ...
    targ_vel_masks = {
        incr_it_right_endpoints[k]: mask for k, mask in enumerate(targ_traj_mask_lists)
    }

    def get_q_pos_and_vel_data(joint_targs_file):
        q_pos_data = get_data_from_qtarg_file(joint_targs_file, dt)
        q_pos_targs = q_pos_data["targ_val"]
        q_pos_time_tks = q_pos_data["tk"]
        joint_names = q_pos_data["joint_names"]
        # qpos_adrs = [model.joint(n).qposadr.item() for n in joint_names]
        qpos_adrs = [70, 71, 72]
        q_pos_targs_expanded = np.zeros((Tk, model.nq))
        # q_pos_targs_expanded = np.zeros((Tk, model.nq))
        # for tk, targ in zip(q_pos_time_tks, q_pos_targs):
        #     q_pos_targs_expanded[tk][qpos_adrs] = targ
        q_pos_mask_list = masks.make_basic_qpos_masks(
            qpos_adrs,
            incr_time_right_endpoints,
            model.nq,
        )
        # q_pos_mask_list = masks.make_basic_qpos_masks(
        #     q_pos_time_tks,
        #     q_pos_qpos_adrs,
        #     incr_time_right_endpoints,
        #     model.nq,
        # )
        q_pos_mask_dict = {
            it: mask for it, mask in zip(incr_it_right_endpoints, q_pos_mask_list)
        }
        q_vel_mask_list = masks.make_basic_qpos_masks(
            # list(range(0, Tk)),
            list(range(0, model.nv)),
            incr_time_right_endpoints,
            model.nv,
        )
        q_vel_mask_dict = {
            it: mask for it, mask in zip(incr_it_right_endpoints, q_vel_mask_list)
        }
        # This is for velocity penalization (l2 regularization)
        # TODO: add an additional dict to allow for target velocities as well
        # as velocity penalties
        # q_vel_mask_dict = {
        #     it: np.ones_like((Tk, model.nv)) for it in incr_it_right_endpoints
        # }
        q_vel_targs_expanded = np.zeros((Tk, model.nv))
        return (
            q_pos_targs_expanded,
            q_vel_targs_expanded,
            q_pos_mask_dict,
            q_vel_mask_dict,
            qpos_adrs,
            joint_names,
        )

    def make_return_dict(
        traj_targs,
        traj_masks,
        traj_vels,
        traj_vels_masks,
        q_pos_targs,
        q_vel_targs,
        q_pos_masks,
        q_vel_masks,
        ctrl_reg_weights,
    ):
        return dict(  # TODO: make naming consistent
            traj_targs=traj_targs,
            traj_masks=traj_masks,
            vel_targs=traj_vels,
            vel_masks=traj_vels_masks,
            # targ_traj_mask_types=mask_types,
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
            _,
            _,
        ) = get_q_pos_and_vel_data(joint_targs_file)

        rs, thetas, _ = basic_movements.random_arcs_right_arm(
            model, data, Tk, data.site(RHAND_S).xpos, smoothing_time, arc_std, seed
        )
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:, 1] = rs * np.cos(thetas)
        traj1_xs[:, 2] = rs * np.sin(thetas)
        traj1_xs += data.site(RSHOULD_S).xpos
        targ_trajs = [traj1_xs]
        ctrl_reg_weights = [None]
        breakpoint()
        return make_return_dict(
            targ_trajs,
            targ_traj_masks,
            q_pos_targs,
            q_vel_targs,
            q_pos_masks,
            q_vel_masks,
            ctrl_reg_weights,
        )
    elif exp_name == "basic_movements_left":
        joint_targs_file = "exp_configs/basic_movements_left_joint_targs.csv"
        (
            q_pos_targs,
            q_vel_targs,
            q_pos_masks,
            q_vel_masks,
            _,
            _,
        ) = get_q_pos_and_vel_data(joint_targs_file)

        rs, thetas, _ = basic_movements.random_arcs_left_arm(
            model, data, Tk, data.site(LHAND_S).xpos, smoothing_time, arc_std, seed
        )
        traj1_xs = np.zeros((Tk, 3))
        traj1_xs[:, 1] = rs * np.cos(thetas)
        traj1_xs[:, 2] = rs * np.sin(thetas)
        traj1_xs += data.site(LSHOULD_S).xpos
        targ_trajs = [traj1_xs]
        ctrl_reg_weights = [None]
        return make_return_dict(
            targ_trajs,
            targ_traj_masks,
            q_pos_targs,
            q_vel_targs,
            q_pos_masks,
            q_vel_masks,
            ctrl_reg_weights,
        )
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
    elif exp_name == "ball_throw":
        joint_targs_file = "exp_configs/ball_throw_joint_targs.csv"
        (
            q_pos_targs,
            q_vel_targs,
            q_pos_masks,
            q_vel_masks,
            _,
            _,
        ) = get_q_pos_and_vel_data(joint_targs_file)
        out = throw_traj(model, data, Tk)
        traj, vel, time_dict = out

        traj_below = traj - np.array([0, 0, 1])

        targ_vels = [vel, vel]
        targ_trajs = [traj, traj_below]

        # q_targs = [np.zeros((Tk, syssize))]
        # q_targ_mask = np.zeros((Tk, syssize2))
        # q_targ_mask2 = np.zeros((Tk, syssize2))
        # # TODO: resolve quaternion
        # q_targ_mask2[time_dict["t_1"] :, joints["all"]["wrist_left"]] = 1
        # q_targ_nz = np.linspace(0, -2.44, time_dict["t_2"] - time_dict["t_1"])
        # q_targ[time_dict["t_1"] : time_dict["t_2"], joints["all"]["wrist_left"]] = (
        #     q_targ_nz
        # )
        # q_targ[time_dict["t_2"] :, joints["all"]["wrist_left"]] = -2.44
        # q_targ_masks = [q_targ_mask, q_targ_mask2, q_targ_mask, q_targ_mask]
        # q_targ_mask_types = ["const"]
        # q_targs = [q_targ]
        # tkzero =
        for k, it in enumerate(targ_traj_masks):
            if it > grab_phase_it:
                for tk in range(grab_phase_tk):
                    targ_traj_masks[it][tk] = 0
                    targ_vel_masks[it][tk] = 0
                    q_pos_masks[it][tk] = 0
                    q_vel_masks[it][tk] = 0
            Tkk = incr_time_right_endpoints[k]
            for tk in range(0, Tkk - mask_window_tk):
                targ_traj_masks[it][tk] = 0

        ctrl_reg_weights = [None]
        return make_return_dict(
            targ_trajs,
            targ_traj_masks,
            targ_vels,
            targ_vel_masks,
            q_pos_targs,
            q_vel_targs,
            q_pos_masks,
            q_vel_masks,
            ctrl_reg_weights,
        )
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
        ret_vals.append(ret_fn(model, data))
    render_fn()
    for tk in range(Tk):
        util.step(model, data, ctrls[tk])
        if ret_fn is not None:
            ret_vals.append(ret_fn(model, data))
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
    n_steps_adh=40,
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


class LimLowestDict(sc.SortedDict):
    def __init__(self, max_len):
        self.max_len = max_len
        super().__init__()
        # self.dict = sc.SortedDict()

    def append(self, key, val):
        self[key] = val
        if len(self) > self.max_len:
            self.popitem()


def show_plot(
    axs,
    hxs,
    tt,
    target_trajs,
    targ_traj_mask,
    site_names=None,
    site_grad_idxs=None,
    ctrls=None,
    losses=None,
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
        tm = np.tile(targ_traj_mask[k] > 0, (3, 1)).T
        tm[tm == 0] = np.nan
        ft = target_trajs[k] * tm
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
            # ax.plot(tt[:-1], ctrls[:, site_grad_idxs[k]])
            for id in site_grad_idxs[k]:
                ax.plot(tt[:-1], ctrls[:, id], label=f"{id}")
            ax.set_ylabel("ctrls")
            # ax.legend()
        ax_cntr += 1
    if losses is not None:
        # axs[s, 0].cla()
        for k in nr:
            loss_k = losses[k]
            # axs[2, 0].plot(tt[: tk + 1], loss_site_xposs[0, 0, k0, : tk + 1])
            ax = axs[ax_cntr, k]
            ax.cla()
            ax.plot(range(loss_k.shape[0]), loss_k)
            ax.set_xlabel("it")
            ax.set_ylabel("site loss")
        ax_cntr += 1
    if grads is not None:
        for k in nr:
            ax = axs[ax_cntr, k]
            ax.cla()
            grad = np.zeros((len(tt) - 1, grads[0].shape[1]))
            grad[: grads[k].shape[0]] = grads[k]
            ax.plot(tt[:-1], grad)
            ax.set_ylabel("grads")
        ax_cntr += 1
    # if ctrls is not None:
    # axs[1,0].plot(tt[:-1], ctrls[:, -2])
    # axs[1,1].plot(tt[:-1], ctrls[:, -1])
    if qvals is not None:
        for k in nr:
            ax = axs[ax_cntr, k]
            ax.cla()
            val_new = butil.propagate_singleton_points(qvals[k])
            ax.plot(tt, val_new, linewidth=3)
            ax.set_prop_cycle(None)
            targ_new = butil.propagate_singleton_points(qtargs[k])
            ax.plot(tt, targ_new)
            ax.set_xlim(lims)
            ax.set_ylabel("q_pos")
    fig.tight_layout()
    # if show:
    #     plt.show(block=False)
    #     # plt.show(block=True)
    #     plt.pause(0.05)
    # if save:
    #     fig.savefig("fig.pdf")


def get_last_timepoint(mask):
    """Get index of last nonzero entry in mask."""
    nonzero = np.where(mask)[0]
    if len(nonzero) == 0:
        breakpoint()
    return nonzero[-1].item()


def arm_target_traj(
    config_name,
    env,
    site_names,
    site_grad_idxs,
    stabilize_jnt_idx,
    stabilize_act_idx,
    ctrls,
    grad_trunc_tk,
    seed,
    ctrl_rate,
    ctrl_std,
    Tk,
    max_its=30,
    lrs=[10],
    phase_2_it=None,
    keep_top=1,
    incr_everys=[10],
    mask_window_tk=5,
    tk_incrs=[5],
    grad_update_every=1,
    grab_phase_it=0,
    grab_phase_tk=0,
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
    ctrl_reg_weight=None,
    q_pos_weight=1,
    joint_penalty_factor=0,
    mask_decay_factor=0.9,
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
        incr_everys: number of iterations between mask increments
        tk_incr: number of timesteps to increment the mask by each
            time it is incremented
        ctrls: initial arm controls
        grad_trunc_tk: gradient truncation time
        seed: random seed
        CTRL_RATE: control rate
        CTRL_STD: control standard deviation
        Tk: number of time steps
        max_its: maximum number of gradient steps
        lr: learning rate
        keep_top: number of lowest losses to keep
    """
    if phase_2_it is None:
        phase_2_it = max_its
    if plot_every is None:
        update_plot_every = max_its
    if render_every is None:
        render_every = max_its

    def shift_endpoints(inp_dict):
        keys = [0] + list(inp_dict.keys())
        ret_dict = {}
        for k in range(len(keys) - 1):
            ret_dict[keys[k]] = inp_dict[keys[k + 1]]
        return ret_dict

    model = env.model
    data = env.data
    nu = model.nu
    nv = model.nv
    traj_and_masks = make_traj_sets(
        env,
        config_name,
        Tk,
        tk_incrs,
        incr_everys,
        phase_2_it,
        mask_window_tk,
        seed,
        mask_decay_factor,
        grab_phase_it,
        grab_phase_tk,
    )

    # traj_and_masks["q_pos_masks"] = [
    #     params["joint_penalty_factor"] * x for x in traj_and_masks["q_pos_masks"]
    # ]

    traj_targs = traj_and_masks["traj_targs"]
    traj_masks = butil.LeftEndpointDict(shift_endpoints(traj_and_masks["traj_masks"]))
    vel_targs = traj_and_masks["vel_targs"]
    vel_masks = butil.LeftEndpointDict(shift_endpoints(traj_and_masks["vel_masks"]))
    q_pos_targs = traj_and_masks["q_pos_targs"]
    q_pos_masks = butil.LeftEndpointDict(shift_endpoints(traj_and_masks["q_pos_masks"]))
    q_vel_targs = traj_and_masks["q_vel_targs"]
    q_vel_masks = butil.LeftEndpointDict(shift_endpoints(traj_and_masks["q_vel_masks"]))
    for key in q_vel_masks:
        q_vel_masks[key] = joint_penalty_factor * q_vel_masks[key]

    incr_its = sorted(list(traj_masks.keys()))

    render_class = butil.targetRender(env, traj_targs, site_names)
    render_fn = render_class.render

    not_stabilize_act_idx = [k for k in range(model.nu) if k not in stabilize_act_idx]

    n_sites = len(site_names)

    data0 = copy.deepcopy(data)

    noisev = make_noisev(model, seed, Tk, ctrl_std, ctrl_rate)

    util.reset_state(model, data, data0)

    def ret_fn(model, data):
        # jnt_ids = [55, 56, 57]
        # vel_ids = opt_utils.convert_qdof_adr(model, jnt_ids, True)
        # site_dict = {}
        ret_dict = {}
        for site in site_names:
            ctrl0 = opt_utils.get_ctrl0(
                model, data, list(range(model.njnt)), site_grad_idxs[k]
            )
            ret_dict[site + "_xpos"] = data.site(site).xpos.copy()
            ret_dict[site + "_ctrl0"] = ctrl0

        ret_dict.update(
            {
                # "site_dict": site_dict,
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "ctrl": data.ctrl.copy(),
            }
        )
        mj.mj_inverse(model, data)
        # site_dict.update({"thorax_forces": data.qfrc_inverse[vel_ids].copy()})
        return ret_dict

    ### Gradient descent
    qpos0 = data.qpos.copy()

    dt = model.opt.timestep
    T = Tk * dt
    tt = np.arange(0, T, dt)

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

    optms = [None] * n_sites
    lowest_losses = LimLowestDict(keep_top)
    lowest_losses_curr_mask = LimLowestDict(keep_top)

    nplots = 5
    fig, axs = plt.subplots(nplots, n_sites, figsize=(nplots * n_sites, 4 * 3.5))
    if n_sites == 1:
        axs = axs.reshape((nplots, 1))
    Tk_trunc_prev = 0
    loss_site_xposs = np.zeros((2, len(site_names), max_its, Tk))
    loss_vels = np.zeros((2, len(site_names), max_its, Tk))
    loss_qposs = np.zeros((2, max_its, Tk))
    loss_qvels = np.zeros((2, max_its, Tk))
    loss_ctrls = np.zeros((2, len(site_names), max_its, Tk - 1))
    # ctrl_reg_weight = 0
    lr = lrs[0]

    for k0 in range(max_its):
        if k0 >= phase_2_it:
            lr = lrs[1]
        if k0 in incr_its:
            for k in range(n_sites):
                optms[k] = get_opt(lr)
        progbar.update(" it: " + str(k0))

        traj_mask_curr = np.array(traj_masks[k0])
        vel_mask_curr = 0 * np.array(vel_masks[k0])
        q_pos_mask_curr = np.array(q_pos_masks[k0])
        q_vel_mask_curr = np.array(q_vel_masks[k0])

        Tk_trunc = get_last_timepoint(traj_mask_curr)
        traj_mask_curr = traj_mask_curr[: Tk_trunc + 1]
        vel_mask_curr = vel_mask_curr[: Tk_trunc + 1]
        q_pos_mask_curr = q_pos_mask_curr[: Tk_trunc + 1] * q_pos_weight
        q_vel_mask_curr = q_vel_mask_curr[: Tk_trunc + 1]
        if Tk_trunc_prev > 0 and Tk_trunc != Tk_trunc_prev:
            ctrls = lowest_losses_curr_mask.popitem(0)[1][1]
            lowest_losses_curr_mask = LimLowestDict(keep_top)
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
            tic = time.time()
            grads[k] = opt_utils.traj_deriv_new(
                model,
                data,
                ctrls_trunc + noisev_trunc,
                traj_targs[k][: Tk_trunc + 1],
                traj_mask_curr,
                vel_targs[k][: Tk_trunc + 1],
                vel_mask_curr,
                q_pos_targs[: Tk_trunc + 1],
                q_pos_mask_curr,
                q_vel_targs[: Tk_trunc + 1],
                q_vel_mask_curr,
                grad_trunc_tk,
                deriv_ids=site_grad_idxs[k],
                deriv_site=site_names[k],
                update_every=grad_update_every,
                update_phase=update_phase,
                let_go_times=let_go_times,
                let_go_ids=let_go_ids,
                n_steps_adh=n_steps_adh,
                contact_check_list=contact_check_list,
                adh_ids=adh_ids,
                ctrl_reg_weight=ctrl_reg_weight,
            )
            # grads[k] = grads[k] / np.linalg.norm(grads[k])
            util.reset_state(model, data, data0)
            toc = time.time()
            print(f"grad time: {toc - tic}")
        losses = [0] * n_sites
        for k in range(n_sites):
            ctrls_trunc[:, site_grad_idxs[k]] = optms[k].update(
                ctrls_trunc[:, site_grad_idxs[k]], grads[k], "ctrls", losses[k]
            )
        ret_dict = forward_and_collect_data(env, ctrls_trunc, ret_fn)
        util.reset_state(model, data, data0)
        for k, site_name in enumerate(site_names):
            site_xpos = ret_dict[site_name + "_xpos"]
            site_ctrl0 = ret_dict[site_name + "_ctrl0"]
            site_deriv = np.diff(site_xpos, axis=0, prepend=site_xpos[:1]) / dt
            loss_site_xposs[0, k, k0, : Tk_trunc + 1] = (
                0.5
                * ((site_xpos - traj_targs[k][: Tk_trunc + 1]) ** 2).mean(axis=1)
                * traj_mask_curr
            )
            loss_vels[0, k, k0, : Tk_trunc + 1] = (
                0.5
                * ((site_deriv - vel_targs[k][: Tk_trunc + 1]) ** 2).mean(axis=1)
                * vel_mask_curr
            )
            loss_ctrls[0, k, k0, :Tk_trunc] = (
                0.5
                * ((ctrls_trunc[:, site_grad_idxs[k]] - site_ctrl0[:-1]) ** 2).mean(
                    axis=1
                )
                * ctrl_reg_weight
            )
        loss_qposs[0, k0, : Tk_trunc + 1] = 0.5 * (
            (ret_dict["qpos"] - q_pos_targs[: Tk_trunc + 1]) ** 2 * q_pos_mask_curr
        ).mean(axis=1)
        loss_qvels[0, k0, : Tk_trunc + 1] = 0.5 * (
            (ret_dict["qvel"] - q_vel_targs[: Tk_trunc + 1]) ** 2 * q_vel_mask_curr
        ).mean(axis=1)
        hxs = [ret_dict[site + "_xpos"] for site in site_names]

        try:
            ctrls_trunc, _, qpos, _ = opt_utils.get_stabilized_ctrls(
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
        except np.linalg.LinAlgError:
            print("LinAlgError in get_stabilized_ctrls")
            ctrls_trunc[:, not_stabilize_act_idx] *= 0.99

        ctrls[:Tk_trunc] = ctrls_trunc.copy()
        # tmp[k0] = ctrls[50, site_grad_idxs[0]]
        tk = Tk_trunc
        util.reset_state(model, data, data0)
        render = k0 % render_every == 0
        if env.render_mode == "human" and render:
            ret_dict = forward_and_collect_data(env, ctrls[:tk], ret_fn, render_fn)
            render_class.reset_counter()
        else:
            ret_dict = forward_and_collect_data(env, ctrls[:tk], ret_fn, False)
        ret_dict["trajectory_target"] = traj_targs
        ret_dict["trajectory_mask"] = traj_mask_curr
        ret_dict["site_names"] = site_names
        util.reset_state(model, data, data0)
        with open(f"output/data_{k0}.pkl", "wb") as f:
            pkl.dump(ret_dict, f)
        with open("output/data_latest.pkl", "wb") as f:
            pkl.dump(ret_dict, f)
        # for k, site_name in enumerate(site_names):
        #     site_xpos = ret_dict[site_name + "_xpos"]
        #     site_ctrl0 = ret_dict[site_name + "_ctrl0"]
        #     site_deriv = np.diff(site_xpos, axis=0, prepend=site_xpos[:1]) / dt
        #     loss_site_xposs[1, k, k0, : Tk_trunc + 1] = (
        #         0.5
        #         * ((site_xpos - traj_targs[k][: Tk_trunc + 1]) ** 2).mean(axis=1)
        #         * traj_mask_curr
        #     )
        #     loss_vels[1, k, k0, : Tk_trunc + 1] = (
        #         0.5
        #         * ((site_deriv - vel_targs[k][: Tk_trunc + 1]) ** 2).mean(axis=1)
        #         * vel_mask_curr
        #     )
        #     loss_ctrls[1, k, k0, :Tk_trunc] = (
        #         0.5
        #         * ((ctrls_trunc[:, site_grad_idxs[k]] - site_ctrl0[:-1]) ** 2).mean(
        #             axis=1
        #         )
        #         * ctrl_reg_weight
        #     )
        # loss_qposs[1, k0, : Tk_trunc + 1] = 0.5 * (
        #     (ret_dict["qpos"] - q_pos_targs[: Tk_trunc + 1]) ** 2 * q_pos_mask_curr
        # ).mean(axis=1)
        # loss_qvels[1, k0, : Tk_trunc + 1] = 0.5 * (
        #     (ret_dict["qvel"] - q_vel_targs[: Tk_trunc + 1]) ** 2 * q_vel_mask_curr
        # ).mean(axis=1)
        # breakpoint()
        qpos = ret_dict["qpos"]
        q_targs_masked = []
        qs_list = []
        hxs = [ret_dict[site + "_xpos"] for site in site_names]
        losses_curr_mask = [0] * n_sites
        for k in range(n_sites):
            hx = hxs[k]
            diffsq1 = (hx - traj_targs[k][: tk + 1]) ** 2
            qpos_mask = q_pos_mask_curr[: tk + 1]
            # vel_mask = q_vel_mask_curr[: tk + 1]
            # q_mask = np.hstack((pos_mask, vel_mask))
            losses[k] = np.mean(diffsq1)
            mask = traj_mask_curr[: tk + 1] > 0
            mask_tiled = np.tile(mask, (3, 1)).T
            temp = np.sum(diffsq1 * mask_tiled) / (np.sum(mask))
            losses_curr_mask[k] = temp
            q_targs = q_pos_targs
            # q_targs = np.hstack((q_pos_targs, q_vel_targs))

            q_targs_masked_tmp = q_targs[: tk + 1].copy()
            q_targs_masked_tmp[qpos_mask == 0] = np.nan
            q_targs_masked.append(q_targs_masked_tmp)
            qs_tmp = qpos.copy()
            qs_tmp[qpos_mask == 0] = np.nan
            qs_list.append(qs_tmp)
        loss = sum([loss.item() for loss in losses]) / n_sites
        lowest_losses.append(loss, (k0, ctrls.copy()))
        loss_curr_mask_avg = sum([loss.item() for loss in losses_curr_mask]) / n_sites
        lowest_losses_curr_mask.append(loss_curr_mask_avg, (k0, ctrls.copy()))
        toc = time.time()
        # print(loss, toc-tic)

        if k0 % plot_every == 0:
            # qs_wr = qs[:, joints['all']['wrist_left']]
            # print()
            # print(ctrls[:10, :5])
            # print()
            # print(ctrls_trunc[:10, :5])
            # print()
            # print(grads[0][:10, :5])
            # print()
            show_plot(
                axs,
                hxs,
                tt[: tk + 1],
                [x[: tk + 1] for x in traj_targs],
                [traj_mask_curr[: tk + 1]],
                # qs_wr,
                # q_targs_wr,
                site_names,
                site_grad_idxs,
                ctrls[:tk],
                qvals=qs_list,
                qtargs=q_targs_masked,
                losses=loss_site_xposs[0, :, :k0, : tk + 1].mean(axis=-1),
                grads=grads,
                # qs_list,
                # q_targs_masked,
                show=False,
            )
            # s = 3
            # axs[s, 0].cla()
            # # axs[2, 0].plot(tt[: tk + 1], loss_site_xposs[0, 0, k0, : tk + 1])
            # axs[s, 0].plot(range(k0), loss_site_xposs[0, 0, :k0, : tk + 1].mean(axis=1))
            # axs[s, 0].set_xlabel("it")
            # axs[s, 0].set_ylabel("site loss")
            # plt.pause(0.1)
            if k0 == 0:
                # Plot again to refresh the window so it resizes to a proper size
                show_plot(
                    axs,
                    hxs,
                    tt[: tk + 1],
                    [x[: tk + 1] for x in traj_targs],
                    [traj_mask_curr[: tk + 1]],
                    # qs_wr,
                    # q_targs_wr,
                    site_names,
                    site_grad_idxs,
                    ctrls[:tk],
                    qvals=qs_list,
                    qtargs=q_targs_masked,
                    grads=grads,
                    # qs_list,
                    # q_targs_masked,
                    show=False,
                )
                # plt.pause(0.1)
            fig.savefig(f"output/fig_{k0}.pdf")
            fig.savefig("output/fig_latest.pdf")
        # util.reset_state(model, data, data0)
        # ctrls = forward_with_dynamic_adhesion(env, ctrls, noisev, True)
        # plt.show()
        # if k0 > phase_2_it:

        # util.reset_state(model, data, data0)
        # hx = forward_with_site(env, ctrls, site_names[0], True)
        Tk_trunc_prev = Tk_trunc
    # except KeyboardInterrupt:
    # pass

    return ctrls, lowest_losses
