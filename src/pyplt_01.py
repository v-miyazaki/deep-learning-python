import numpy as np
import matplotlib.pyplot as plt

def perceptron(x, w, b):
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def AND(x1, x2):
    return perceptron(x=np.array([x1, x2]), w=np.array([0.5, 0.5]), b=-0.7)

def OR(x1, x2):
    return perceptron(x=np.array([x1, x2]), w=np.array([0.5, 0.5]), b=-0.2)

def NAND(x1, x2):
    return perceptron(x=np.array([x1, x2]), w=np.array([-0.5, -0.5]), b=0.7)

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == "__main__":
    print(XOR(0, 0), XOR(0, 1), XOR(1, 0), XOR(1, 1))

    # plt.title(__file__)
    # plt.grid(True)

    # # x1 = np.linspace(0, 1)
    # # x2 = np.linspace(0, 1)
    # # y = b

    # b = -0.5
    # x1 = np.array([0, 1])
    # x2 = -b - x1
    # # plt.plot(x1, x2)
    # plt.fill_between(x1, x2, x2.min())

    # plt.show()
    pass