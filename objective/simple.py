import numpy as np

Y = '_y_column'

class Objective:
    def __init__(self, lambda_value=1, weights={0.: 1., 1.: 1.}):
        self.lambda_1 = lambda_value
        """
        Due to optimization weights value doesn't affect result at the moment.
        Dependency on weights should be implemented later
        """
        self.weights = weights

    def _g_func(self, val_target, val_y):
        return self.weights.get(val_target) * 2 * (val_y - val_target)


    def _h_func(self, val_target):
        return self.weights.get(val_target) * 2

    def _get_g(self, data, idx_target, idx_y):
        # return sum(np.vectorize(self._g_func)(data[target], data[Y]))
        return 2 * (np.sum(data[:, idx_y]) - np.sum(data[:, idx_target]))


    def _get_h(self, data, idx_target, idx_y):
        # return sum(np.vectorize(self._h_func)(data[target]))
        return 2

    def _get_obj(self, g_lo, h_lo, g_hi, h_hi):
        return -0.5 * (g_lo * g_lo / (h_lo + self.lambda_1) + g_hi * g_hi / (h_hi + self.lambda_1)), \
               self._get_w(g_lo, h_lo), \
               self._get_w(g_hi, h_hi)

    def get_objective(self, lo_data, hi_data, idx_target, idx_y):
        return self._get_obj(self._get_g(lo_data, idx_target, idx_y), self._get_h(lo_data, idx_target, idx_y),
                        self._get_g(hi_data, idx_target, idx_y), self._get_h(hi_data, idx_target, idx_y))

    def _get_w(self, g, h):
        return -g / (h + self.lambda_1)