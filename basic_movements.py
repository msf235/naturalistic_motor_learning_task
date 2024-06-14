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
import sortedcontainers as sc
from matplotlib import pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def reflective_random_walk(n_steps=1000, initial_position=0.5, step_std=0.02,
                           smoothing_sigma=10, lower_lim=0, upper_lim=1):
    # n_steps: Number of steps in the random walk
    # initial_position: Starting position of the random walk
    # step_std: Standard deviation of the increments
    # smoothing_sigma: Sigma value for Gaussian smoothing
    
    # Initialize the random walk array
    positions = np.zeros(n_steps)
    positions[0] = initial_position
    
    # Generate random steps
    steps = np.random.normal(loc=0, scale=step_std, size=n_steps)
    
    # Perform the random walk with reflective boundary conditions
    for i in range(1, n_steps):
        new_position = positions[i-1] + steps[i]
        if new_position < lower_lim:
            new_position = -new_position  # Reflect off the lower boundary
        elif new_position > upper_lim:
            new_position = 2*upper_lim - new_position  # Reflect off the upper boundary
        positions[i] = new_position
    
    # Smooth the positions using Gaussian filter
    smoothed_positions = gaussian_filter1d(positions, sigma=smoothing_sigma,
                                           mode='nearest')
    
    return positions, smoothed_positions

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

def random_arcs(shouldx, handx, elbowx, n_steps, initial_xpos,
                theta_lims, smoothing_sigma=20, step_std=0.02):
    r1 = np.sum((shouldx - elbowx)**2)**.5
    r2 = np.sum((elbowx - handx)**2)**.5
    r = r1 + r2

    init_pos_rel = initial_xpos-shouldx
    r0 = (init_pos_rel**2).sum()**0.5
    th0 = np.arctan2(init_pos_rel[2], init_pos_rel[1])
    if th0 < -np.pi/2:
        th0 += 2*np.pi

    # Random walk for radius
    positions, smoothed_positions = reflective_random_walk(
        n_steps=n_steps, initial_position=r0, step_std=step_std,
        smoothing_sigma=smoothing_sigma, lower_lim=0, upper_lim=r
    )

    rs = smoothed_positions - smoothed_positions[0] + r0

    # Random walk for angles
    positions, smoothed_positions = reflective_random_walk(
        n_steps=n_steps, initial_position=th0, step_std=step_std,
        smoothing_sigma=smoothing_sigma, lower_lim=theta_lims[0],
        upper_lim=theta_lims[1]
    )
    thetas = smoothed_positions - smoothed_positions[0] + th0

    return rs, thetas

def random_arcs_right_arm(model, data, n_steps, initial_xpos,
                          smoothing_sigma=20, step_std=0.02):
    shouldx = data.site('shoulder1_right').xpos
    elbowx = data.site('elbow_right').xpos
    handx = data.site('hand_right').xpos
    theta_max = 1.2*np.pi
    theta_min = np.pi/2.5
    if smoothing_sigma is None:
        t_sm = .1
        smoothing_sigma = int(t_sm / model.opt.timestep)

    rs, thetas = random_arcs(shouldx, handx, elbowx, n_steps, initial_xpos,
                             (theta_min, theta_max), smoothing_sigma, step_std)

    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    # plt.plot(xs+shouldx[1], ys+shouldx[2])
    # plt.scatter(shouldx[1], shouldx[2], color='red')
    # plt.scatter(handx[1], handx[2], color='blue')
    # plt.axis('equal')
    # plt.show()

    # Random walk for wrist
    positions, wrist_qs = reflective_random_walk(
        n_steps=n_steps, initial_position=0, step_std=0.02,
        smoothing_sigma=20, lower_lim=-2.44, upper_lim=1.48
    )

    return rs, thetas, wrist_qs

def random_arcs_left_arm(model, data, n_steps, initial_xpos,
                         smoothing_time=None, step_std=0.02):
    shouldx = data.site('shoulder1_left').xpos
    elbowx = data.site('elbow_left').xpos
    handx = data.site('hand_left').xpos
    theta_min = -np.pi/4
    theta_max = np.pi/2.5 + np.pi/2
    if smoothing_time is None:
        smoothing_time = .1
    smoothing_sigma = int(smoothing_time / model.opt.timestep)
    
    rs, thetas = random_arcs(shouldx, handx, elbowx, n_steps, initial_xpos,
                             (theta_min, theta_max), smoothing_sigma, step_std)

    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)
    # plt.plot(xs+shouldx[1], ys+shouldx[2])
    # plt.scatter(shouldx[1], shouldx[2], color='red')
    # plt.scatter(handx[1], handx[2], color='blue')
    # plt.axis('equal')
    # plt.show()

    positions, wrist_qs = reflective_random_walk(
        n_steps=n_steps, initial_position=0, step_std=0.02,
        smoothing_sigma=20, lower_lim=-2.44, upper_lim=1.48
    )

    return rs, thetas, wrist_qs


