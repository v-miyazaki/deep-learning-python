import matplotlib.pyplot as plt
import numpy as np

samples = 100000

# 平均0、分散0.05の正規分布
norm = np.random.normal(0, 0.05**2, samples)
print(norm)
# 一様分布
rand = (np.random.rand(samples) - 0.5) * 0.01
print(rand)

# plt.plot(norm)
plt.hist(norm, bins=int(samples/100), color="blue")
plt.hist(rand, bins=int(samples/100), color="red")
# plt.scatter(norm[0], norm[1], s=1, c="blue")
# plt.scatter(rand[0], rand[1], s=1, c="red")
plt.show()
