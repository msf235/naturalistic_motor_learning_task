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
    smoothed_positions = gaussian_filter1d(positions, sigma=smoothing_sigma)
    
    return positions, smoothed_positions

# # Plot the results
# plt.figure(figsize=(12, 6))
# plt.plot(positions, label='Original Random Walk', alpha=0.5)
# plt.plot(smoothed_positions, label='Gaussian Smoothed Random Walk', linewidth=2)
# plt.title('Reflective Gaussian Smoothed Random Walk')
# plt.xlabel('Time Step')
# plt.ylabel('Position')
# plt.legend()
# plt.grid(True)
# plt.show()


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

def random_arcs_right_arm(model, data, n_steps, initial_xpos):
    shouldx = data.site('shoulder1_right').xpos
    elbowx = data.site('elbow_right').xpos
    handx = data.site('hand_right').xpos
    ballx = data.site('ball_base').xpos
    r1 = np.sum((shouldx - elbowx)**2)**.5
    r2 = np.sum((elbowx - handx)**2)**.5
    r = r1 + r2

    x = r * cos (theta)
    r = x^2 + y^2
    r0 = ((initial_xpos**2).sum())**0.5
    th0 = np.arctan2(initial_xpos[1], initial_xpos[0])

    # Number of steps in the random walk

    # Random walk for radius
    positions, smoothed_positions = reflective_random_walk(
        n_steps=n_steps, initial_position=r0, step_std=0.02,
        smoothing_sigma=20, lower_lim=0, upper_lim=r
    )

    rs = smoothed_positions * r

    # Random walk for angles
    positions, smoothed_positions = reflective_random_walk(
        n_steps=n_steps, initial_position=0.5, step_std=0.02,
        smoothing_sigma=20
    )
    theta_max = np.pi
    theta_min = np.pi/2.5

    thetas = smoothed_positions * (theta_max-theta_min) + theta_min

    # xs = rs * np.cos(thetas)
    # ys = rs * np.sin(thetas)
    # plt.plot(xs, ys)
    # plt.axis('equal')
    # plt.show()

    return rs, thetas


# def random_arcs_right_arm(model, data, Tk):

def random_arcs(model, data, Tk):
    shouldx = data.site('shoulder1_right').xpos
    elbowx = data.site('elbow_right').xpos
    handx = data.site('hand_right').xpos
    ballx = data.site('ball_base').xpos
    r1 = np.sum((shouldx - elbowx)**2)**.5
    r2 = np.sum((elbowx - handx)**2)**.5
    r = r1 + r2
