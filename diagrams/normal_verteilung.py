import matplotlib.pyplot as plt
import numpy as np
from statistics import median

offspring_1 = 2000
offspring_2 = 5000

between = offspring_2 - offspring_1

min_x = 0
max_x = 10000

variance_factor = 2

expectation = median([offspring_1, offspring_2])
deviation = expectation/max_x

rng = np.random.default_rng()
sample = rng.normal(expectation, between/variance_factor, size=10000)
plt.hist(sample, 100, density=True, color='orange')
plt.axvline(x=deviation*max_x, color='red', linestyle='--', label='deviation')
plt.axvline(x=offspring_1, color='green', linestyle='-', label='offspring1')
plt.axvline(x=offspring_2, color='purple', linestyle='-', label='offspring2')
plt.xlim(min_x, max_x)
plt.legend()
plt.show()