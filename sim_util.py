import numpy as np
import mujoco as mj
import copy

## Reset and burn in:
def reset(model, data, nsteps, humanoid_x0=None):
    jid = model.joint('x_root').jntid
    mj.mj_resetData(model, data)
    if humanoid_x0 is not None:
        data.qpos[jid] = humanoid_x0
    for k in range(nsteps):
        mj.mj_step(model, data)

def reset_state(data_from, data_to):
    """Resets the state of `data_to` to that of `data_from`."""
    data_from.qpos[:] = data_to.qpos.copy()
    data_from.qvel[:] = data_to.qvel.copy()
    data_from.qacc[:] = data_to.qacc.copy()
    data_from.act[:] = data_to.act.copy()
    data_from.ctrl[:] = data_to.ctrl.copy()
    data_from.time = data_to.time
    # data_from.qfrc_applied[:] = data_to.qfrc_applied.copy()
    # state = get_state(data_from)
    # set_state(data_to, state)

def step(model, data, ctrl):
    mj.mj_step1(model, data)
    data.ctrl[:] = ctrl
    mj.mj_step2(model, data)

def get_contact_pairs(model, data):
    contact_pairs = [[model.geom(c.geom[0]).name, model.geom(c.geom[1]).name]
                     for c in data.contact]
    # contact_pairs = np.stack(contact_pairs)
    return contact_pairs

class FilteredNoise:
    def __init__(self, ind_dim, kernel, rng):
        # self.perturb = np.random.randn(ind_dim, len(kernel))
        self.perturb = rng.standard_normal((ind_dim, len(kernel)))
        self.ind_dim = ind_dim
        self.kernel = kernel
        self.rng = rng

    def sample_one(self):
        perturb_smoothed = self.perturb @ self.kernel
        self.perturb[:] = np.roll(self.perturb, -1, axis=1)
        self.perturb[:, -1] = self.rng.standard_normal(self.ind_dim)
        return perturb_smoothed

    def sample(self, nsamples=1):
        samples = []
        for k in range(nsamples):
            samples.append(self.sample_one())
        return np.stack(samples)

    def reset(self, rng):
        self.rng = rng


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
