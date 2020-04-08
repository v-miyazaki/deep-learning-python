import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-np.pi, np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
