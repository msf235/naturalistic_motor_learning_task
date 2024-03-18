import humanoid2d as h2d
import opt_utils as opt_utils
import numpy as np
import sim_utils as util
import mujoco as mj
import sys

### Set things up
seed = 2
rng = np.random.default_rng(seed)
nu = 4

Tk = 50

# Get noise
CTRL_STD = .05       # actuator units
# CTRL_STD = 0       # actuator units
CTRL_RATE = 0.8       # seconds
width = int(CTRL_RATE/.005)
kernel = np.exp(-0.5*np.linspace(-3, 3, width)**2)
kernel /= np.linalg.norm(kernel)
# noise = util.FilteredNoise(nu, kernel, rng)
noise = util.MinimalNoise(rng)
# noisev = CTRL_STD * noise.sample(Tk-1)
noisev = CTRL_STD * noise.sample()
print()
# print(noisev[0])
print(noisev)
print()
sys.exit()

print("test")
