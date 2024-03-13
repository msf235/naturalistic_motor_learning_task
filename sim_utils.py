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
               fixed_joint_inds=[]):
    # WARNING: changes data!
    nqf = model.nq - len(fixed_joint_inds)
    extra_inds = [i+model.nq for i in fixed_joint_inds]
    fixed_joint_inds_2 = fixed_joint_inds + extra_inds
    Tk = qs.shape[0]
    As = np.zeros((Tk, 2*nqf, 2*nqf))
    Bs = np.zeros((Tk, 2*nqf, model.nu))
    # Cs = np.zeros((Tk, 3, nvf))
    

    for tk in range(Tk):
        data.qpos[:] = qs[tk]
        data.qvel[:] = vs[tk]
        data.ctrl[:] = us[tk]
        epsilon = 1e-6
        A = np.zeros((2*model.nq, 2*model.nq))
        B = np.zeros((2*model.nq, model.nu))
        mj.mjd_transitionFD(model, data, epsilon, True, A, B, None,
                            None)
        A = np.delete(A, fixed_joint_inds_2, axis=0)
        As[tk] = np.delete(A, fixed_joint_inds_2, axis=1)
        B = np.delete(B, fixed_joint_inds_2, axis=0)
        Bs[tk] = np.delete(B, fixed_joint_inds, axis=1)
        # mj.mj_jacSite(model, data, Cs[tk], None, site=model.site('').id)

    lams = np.zeros((Tk, 2*nqf))
    # dldq = Cs[Tk-1].T @ dlds
    # lams[Tk-1,:nv] = targ_factor * dldq
    # lams[Tk-1,2] += dldtheta
    lams[Tk-1,:nqf] = lams_fin
    grads = np.zeros((Tk, model.nu))
    tau_loss_factor = 1e-9

    for tk in range(2, Tk):
        lams[Tk-tk] = As[Tk-tk].T @ lams[Tk-tk+1]
        grads[Tk-tk] = (tau_loss_factor/tk**.5)*losses[Tk-tk] \
                + Bs[Tk-tk].T @ lams[Tk-tk+1]
    return grads

