import matplotlib.pyplot as plt
import numpy as np
from statistics import median

offspring_1 = 0.007550123131506232
offspring_2 = 0.024

between = offspring_2 - offspring_1

min_x = 0.00001
max_x = 0.1

variance_factor = 1.2

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


#