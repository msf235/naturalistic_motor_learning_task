import numpy as np
import mujoco as mj

class FilteredNoise:
    def __init__(self, ind_dim, kernel, seed=23):
        self.perturb = np.random.randn(ind_dim, len(kernel))
        self.ind_dim = ind_dim
        self.kernel = kernel
        self.rng = np.random.default_rng(seed)

    def sample(self):
        perturb_smoothed = self.perturb @ self.kernel
        self.perturb[:] = np.roll(self.perturb, -1, axis=1)
        self.perturb[:, -1] = self.rng.standard_normal(self.ind_dim)
        return perturb_smoothed

def traj_deriv(model, data, qs, vs, us, lams_fin, losses,
               fixed_act_inds=[]):
    nufree = model.nu - len(fixed_act_inds)
    # WARNING: changes data!
    Tk = qs.shape[0]
    As = np.zeros((Tk, 2*model.nv, 2*model.nv))
    Bs = np.zeros((Tk, 2*model.nv, nufree))
    B = np.zeros((2*model.nv, model.nu))
    # Cs = np.zeros((Tk, 3, model.nv))
    lams = np.zeros((Tk, 2*model.nv))

    for tk in range(Tk):
        data.qpos[:] = qs[tk]
        data.qvel[:] = vs[tk]
        data.ctrl[:] = us[tk]
        epsilon = 1e-6
        mj.mjd_transitionFD(model, data, epsilon, True, As[tk], B, None,
                            None)
        Bs[tk] = np.delete(B, fixed_act_inds, axis=1)
        # mj.mj_jacSite(model, data, Cs[tk], None, site=model.site('').id)

    # dldq = Cs[Tk-1].T @ dlds
    # lams[Tk-1,:nv] = targ_factor * dldq
    # lams[Tk-1,2] += dldtheta
    lams[Tk-1,:model.nv] = lams_fin
    grads = np.zeros((Tk, nufree))
    tau_loss_factor = 1e-9

    for tk in range(2, Tk):
        lams[Tk-tk] = As[Tk-tk].T @ lams[Tk-tk+1]
        grads[Tk-tk] = (tau_loss_factor/tk**.5)*losses[Tk-tk] \
                + Bs[Tk-tk].T @ lams[Tk-tk+1]
    return grads

