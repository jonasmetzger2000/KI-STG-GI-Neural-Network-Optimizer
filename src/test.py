import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng()

s = rng.normal(10, 5, 1000)
print(s)
plt.hist(s, 30, density=True)
plt.ticklabel_format(style='plain')
plt.show()
