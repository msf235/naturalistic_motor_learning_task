import numpy as np
import mujoco as mj

## Reset and burn in:
def reset(model, data, nsteps):
    mj.mj_resetData(model, data)
    for k in range(nsteps):
        mj.mj_step(model, data)

class FilteredNoise:
    def __init__(self, ind_dim, kernel, rng, std=1.0):
        self.perturb = np.random.randn(ind_dim, len(kernel))
        self.ind_dim = ind_dim
        self.kernel = kernel
        self.rng = rng
        # print(self.rng.standard_normal(3))
        self.std = std

    def sample_one(self):
        perturb_smoothed = self.perturb @ self.kernel
        self.perturb[:] = np.roll(self.perturb, -1, axis=1)
        self.perturb[:, -1] = self.rng.standard_normal(self.ind_dim)
        return self.std*perturb_smoothed

    def sample(self, nsamples=1):
        return np.stack([self.sample_one() for k in range(nsamples)])

    def reset(self, rng):
        self.rng = rng

class BlankNoise:
    def sample(self):
        return 0

def traj_deriv(model, data, qs, vs, us, lams_fin, losses,
               fixed_act_inds=[]):
    nufree = model.nu - len(fixed_act_inds)
    # WARNING: changes data!
    Tk = qs.shape[0]
    As = np.zeros((Tk-1, 2*model.nv, 2*model.nv))
    Bs = np.zeros((Tk-1, 2*model.nv, nufree))
    B = np.zeros((2*model.nv, model.nu))
    # Cs = np.zeros((Tk, 3, model.nv))
    lams = np.zeros((Tk, 2*model.nv))
    not_fixed_act_inds = [i for i in range(model.nu) if i not in
                          fixed_act_inds]

    for tk in range(Tk-1):
        data.qpos[:] = qs[tk]
        data.qvel[:] = vs[tk]
        data.ctrl[:] = us[tk]
        epsilon = 1e-6
        mj.mjd_transitionFD(model, data, epsilon, True, As[tk], B, None, None)
        Bs[tk] = np.delete(B, fixed_act_inds, axis=1)
        # mj.mj_jacSite(model, data, Cs[tk], None, site=model.site('').id)

    lams[Tk-1, :model.nv] = lams_fin
    grads = np.zeros((Tk, nufree))
    # tau_loss_factor = 1e-9
    tau_loss_factor = 0
    loss_u = np.delete(us, fixed_act_inds, axis=1)

    for tk in range(2, Tk):
        lams[Tk-tk] = As[Tk-tk].T @ lams[Tk-tk+1]
        # grads[Tk-tk] = (tau_loss_factor/tk**.5)*loss_u[Tk-tk] \
        grads[Tk-tk] = tau_loss_factor*loss_u[Tk-tk] \
                + Bs[Tk-tk].T @ lams[Tk-tk+1]
    # breakpoint()
    return grads

