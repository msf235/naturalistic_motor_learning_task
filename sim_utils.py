import numpy as np
import mujoco as mj
import copy

## Reset and burn in:
def reset(model, data, nsteps):
    mj.mj_resetData(model, data)
    for k in range(nsteps):
        mj.mj_step(model, data)

class MinimalNoise:
    def __init__(self, rng):
        self.rng = rng

    def sample(self):
        return self.rng.standard_normal()


class FilteredNoise:
    def __init__(self, ind_dim, kernel, rng):
        self.perturb = np.random.randn(ind_dim, len(kernel))
        self.ind_dim = ind_dim
        self.kernel = kernel
        self.rng = rng
        # print(self.rng.standard_normal(3))

    def sample_one(self):
        perturb_smoothed = self.perturb @ self.kernel
        self.perturb[:] = np.roll(self.perturb, -1, axis=1)
        self.perturb[:, -1] = self.rng.standard_normal(self.ind_dim)
        return perturb_smoothed

    def sample(self, nsamples=1):
        return np.stack([self.sample_one() for k in range(nsamples)])

    def reset(self, rng):
        self.rng = rng

class BlankNoise:
    def sample(self):
        return 0


def forward_sim(model, data, ctrls):
    Tk = ctrls.shape[0]
    qs = np.zeros((Tk+1, model.nq))
    qs[0] = data.qpos.copy()
    vs = np.zeros((Tk+1, model.nq))
    vs[0] = data.qvel.copy()
    for k in range(Tk):
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrls[k]
        mj.mj_step2(model, data)
        qs[k+1] = data.qpos.copy()
        vs[k+1] = data.qvel.copy()
    return qs, vs
