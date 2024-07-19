import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

rng = np.random.default_rng()
n, p, size = 10, 0.7, 10000
s = rng.binomial(n, p, 10000)
sample = rng.binomial(n, p, size=size)
count, bins, _ = plt.hist(sample, 30, density=True)
x = np.arange(n)
y = binom.pmf(x, n, p)
plt.plot(x, y, linewidth=2, color='r')
plt.show()
