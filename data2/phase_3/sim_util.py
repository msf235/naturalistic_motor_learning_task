import numpy as np
import mujoco as mj
import copy
import time
import sys

def format_time(time_in_seconds):
    if time_in_seconds < 60:
        return f"{int(time_in_seconds)} seconds"
    elif time_in_seconds < 3600:
        # Convert seconds to minutes, seconds format
        minutes = int(time_in_seconds // 60)
        seconds = int(time_in_seconds % 60)
        return f"{minutes} minutes, {seconds} seconds"
    else:
        # Convert seconds to hours, minutes, seconds format
        hours = int(time_in_seconds // 3600)
        time_in_seconds = time_in_seconds % 3600
        minutes = int(time_in_seconds // 60)
        seconds = int(time_in_seconds % 60)
        return f"{hours} hours, {minutes} minutes, {seconds} seconds"


class ProgressBar:
    def __init__(self, update_every=2, final_it=100):
        tic = time.time()
        self.first_time = tic
        self.latest_time = tic
        self.update_every = update_every
        self.it = 0
        self.final_it = final_it

    def update(self, extra_str=""):
        tic = time.time()
        if tic - self.latest_time > self.update_every:
            elapsed = tic - self.first_time
            frac = (self.it + 1) / self.final_it
            est_time_remaining = elapsed * (1/frac - 1)
            sys.stdout.write('\r\r')
            pstring = "[%-15s] %d%%" % ('='*int(15*frac), 100*frac,)
            pstring += "  Est. time remaining: " \
                        + format_time(est_time_remaining)
            pstring += extra_str
            # Pad with whitespace
            if len(pstring) < 90:
                pstring += ' '*(90-len(pstring))
            sys.stdout.write(pstring)
            sys.stdout.flush()
            self.latest_time = tic
        self.it += 1

## Reset and burn in:
# def reset(model, data, nsteps, humanoid_x0=None):
    # jid = model.joint('human_x_root').jntid
    # mj.mj_resetData(model, data)
    # mj.mj_forward(model, data)
    # if humanoid_x0 is not None:
        # data.qpos[jid] = humanoid_x0
    # for k in range(nsteps):
        # mj.mj_step(model, data)


def reset_state(model, data_to, data_from):
    """Resets the state of `data_to` to that of `data_from`."""
    data_to.qpos[:] = data_from.qpos.copy()
    data_to.qvel[:] = data_from.qvel.copy()
    data_to.qacc[:] = data_from.qacc.copy()
    data_to.act[:] = data_from.act.copy()
    data_to.ctrl[:] = data_from.ctrl.copy()
    data_to.time = data_from.time
    mj.mj_forward(model, data_to)
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
    ss = np.zeros((Tk+1, data.sensordata.shape[0]))
    ss[0] = data.sensordata[:].copy()
    for k in range(Tk):
        mj.mj_step1(model, data)
        data.ctrl[:] = ctrls[k]
        mj.mj_step2(model, data)
        qs[k+1] = data.qpos.copy()
        vs[k+1] = data.qvel.copy()
        ss[k+1] = data.sensordata[:].copy()
    return qs, vs, ss
