import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.exp(-x)
plt.plot(x, y, label="exp(-x)")

x = np.arange(-5, 5, 0.1)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y, label="sigmoid(x)")

x = np.arange(-5, 5, 0.1)
y = 1 / np.exp(-x)
plt.plot(x, y, label="1/exp(-x)")

x = np.arange(-5, 5, 0.1)
y = 1 + np.exp(-x)
plt.plot(x, y, label="1 + exp(-x)")

plt.title("plot")
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
plt.legend(loc = 'upper right')
plt.grid()
plt.show()
