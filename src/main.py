import matplotlib.pyplot as plt
import numpy as np
from statistics import median

offspring_1 = 1000
offspring_2 = 8000

between = offspring_2 - offspring_1

min_x = 0
max_x = 10000

expectation = median([offspring_1, offspring_2])
deviation = expectation/max_x

rng = np.random.default_rng()
sample = rng.normal(expectation, between/4, size=1000)
plt.hist(sample, 100, density=True)
plt.axvline(x=deviation*max_x, color='lime', linestyle='--', label='deviation')
plt.axvline(x=offspring_1, color='green', linestyle='--', label='offspring1')
plt.axvline(x=offspring_2, color='purple', linestyle='--', label='offspring2')
plt.xlim(min_x, max_x)
plt.legend()
plt.show()


# plt.plot(np.arange(max_x), binom.pmf(np.arange(max_x), max_x, median([offspring_1, offspring_2])/max_x))