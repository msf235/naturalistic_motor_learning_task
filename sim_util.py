import numpy as np

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
