from statistics import median

import numpy as np


class Gaus:
    def __init__(self, _from, _to, variance_factor):
        self._from = _from
        self._to = _to
        self.variance_factor = variance_factor
        self.rng = np.random.default_rng()

    def compute(self, x1, x2):
        expectation = median([x1, x2])
        between = max(x1, x2) - min(x1, x2)
        sample = self.rng.normal(expectation, between/self.variance_factor, size=1)
        return max(self._from, min(self._to, float(sample[0])))


